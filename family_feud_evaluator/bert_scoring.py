from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
from copy import deepcopy
from functools import partial
from typing import *

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    BertConfig,
    BertTokenizer,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
    BertModel,
)
from transformers import (
    GPT2Config,
    OpenAIGPTConfig,
    XLNetConfig,
    TransfoXLConfig,
    XLMConfig,
    CTRLConfig,
)

from .evaluation import general_eval
from .scoring import all_pairs_scores

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            GPT2Config,
            OpenAIGPTConfig,
            XLNetConfig,
            TransfoXLConfig,
            XLMConfig,
            CTRLConfig,
        )
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer),
}


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def prepare_input(text, length, tokenizer):
    marked_text = "[CLS] " + text + " [SEP]"
    while len(marked_text.split(" ")) < length + 2:
        marked_text += " [PAD]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    return indexed_tokens


def transform_question(question):
    question = question.lower()
    question = question.replace(".", "")
    question = question.replace(":", "")
    question = question.replace("?", "")
    question = question.replace("someone", "one person")
    question = question.replace("someplace", "one place")
    if "name something" in question:
        question = question.replace("name something", "one thing")
        question += " is"
    elif "tell me something" in question:
        question = question.replace("tell me something", "one thing")
        question += " is"
    elif "name a " in question:
        question = question.replace("name a ", "one ")
        question += " is"
    elif "name an " in question:
        question = question.replace("name an ", "one ")
        question += " is"
    elif "name" in question:
        question = question.replace("name", "")
        question += " is"
    elif question.startswith("tell me a "):
        question = question.replace("tell me a ", "one ")
        question += " is"
    elif question.startswith("tell me an "):
        question = question.replace("tell me an ", "one ")
        question += " is"
    elif question.startswith("what "):
        question = question.replace("what", "one")
        question += " is"
    elif question.startswith("give me a "):
        question = question.replace("give me a ", "one ")
        question += " is"
    elif question.startswith("tell me "):
        question = question.replace("tell me ", "")
        question += " is"
    elif "which" in question:
        question = question.replace("which", "one")
        question += " is"
    elif "what" in question:
        question = question.replace("what", "one")
        question += " is"
    elif "how can you tell" in question:
        question = question.replace("how can you tell", "one way to tell")
        question += " is"
    else:
        question = "Q: " + question + "? A: "
    return question


class TransformerScoringModel:
    def __init__(
        self,
        model_type="bert",
        model_name_or_path="bert-base-cased",
        no_cuda=True,
        seed=42,
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            torch.cuda.manual_seed_all(seed)
        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        self.tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
        model = model_class.from_pretrained(model_name_or_path)
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
        )
        self.device = device
        model.to(self.device)
        model.eval()
        self.model = model

    def __call__(self, question, answer):
        """
        Returns the average of the contextualized answer embeddings.
        """
        question = transform_question(question)
        text = question + " " + answer

        question_length = len(self.tokenizer.tokenize(question))
        answer_length = len(self.tokenizer.tokenize(answer))
        indexed_tokens = prepare_input(text, len(text.split(" ")), self.tokenizer)

        tokens_tensor = torch.tensor([indexed_tokens], device=self.device)
        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor)

        # take the average of the answer embeddings.
        representation = encoded_layers[
            0, question_length : question_length + answer_length, :
        ].mean(dim=0)
        return representation

    def preprocessing(
        self, q: Dict[str, Union[str, Dict]], answers: List[str]
    ) -> Tuple[Dict[str, Union[str, Dict]], List[str]]:
        """ Pre-processes a given question to replace the true answers with their vector representations
        :param q: A dict containing keys 'question' and 'answers-cleaned'.
        :return: The same dict, where the 'answers-cleaned' have been replaced by vector representations of themselves.
        """
        q = self._preprocess_true_answers(q)
        answers = self._preprocess_pred_answers(q["question"], answers)
        return q, answers

    def _preprocess_pred_answers(
        self, question: str, answers: List[str]
    ) -> List[torch.Tensor]:
        return [self(question, ans) for ans in answers]

    def _preprocess_true_answers(
        self, q: Dict[str, Union[str, Dict]]
    ) -> Dict[str, Union[str, Dict]]:
        q = deepcopy(q)
        answers_cleaned_out = dict()
        question = q["question"]
        for ans, count in q["answers-cleaned"].items():
            if isinstance(ans, frozenset):
                ans_out = frozenset({self(question, a) for a in ans})
            else:
                ans_out = self(question, ans)
            answers_cleaned_out[ans_out] = count
        q["answers-cleaned-orig"] = q["answers-cleaned"]
        q["answers-cleaned"] = answers_cleaned_out
        return q


@torch.no_grad()
def cosine_similarity_score(
    pred_answer: torch.Tensor, true_answer: torch.Tensor
) -> float:
    return ((F.cosine_similarity(pred_answer, true_answer, 0) + 1) / 2).item()


def tensor_cluster_score_func(
    pred_answers: List[str],
    true_answers: Union[Dict[str, int], Dict[frozenset, int]],
    score_func: Callable = cosine_similarity_score,
    cluster_reduction_func: Callable = np.max,
) -> np.ndarray:
    true_ans, *_ = true_answers.keys()
    if isinstance(true_ans, frozenset):
        score_func = partial(
            all_pairs_scores,
            score_func=score_func,
            reduction_func=cluster_reduction_func,
            preprocess_func=lambda z: [z] if isinstance(z, torch.Tensor) else z,
        )
    return all_pairs_scores(pred_answers, true_answers, score_func)


hard_bert_eval = partial(
    general_eval,
    score_func=cosine_similarity_score,
    cluster_score_func=tensor_cluster_score_func,
    string_preprocessing=lambda z: z,
    score_matrix_transformation=np.round,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default="bert",
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-cased",
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    args = parser.parse_args()

    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    original_question = (
        "Name something that is hard to guess about a person you are just meeting."
    )
    question = transform_question(original_question)
    ans1 = "age"
    ans2 = "life experience"
    text1 = question + " " + ans1
    text2 = question + " " + ans2
    length = max(len(text2.split(" ")), len(text1.split(" ")))

    question_length = len(tokenizer.tokenize(question))
    ans1_length = len(tokenizer.tokenize(ans1))
    ans2_length = len(tokenizer.tokenize(ans2))
    indexed_tokens1 = prepare_input(text1, length, tokenizer)
    indexed_tokens2 = prepare_input(text2, length, tokenizer)

    tokens_tensor = torch.tensor([indexed_tokens1, indexed_tokens2], device=args.device)
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor)

    # take the average of the answer embeddings.
    representation1 = encoded_layers[
        0, question_length : question_length + ans1_length, :
    ].mean(dim=0)
    representation2 = encoded_layers[
        1, question_length : question_length + ans2_length, :
    ].mean(dim=0)
    score = F.cosine_similarity(representation1.cpu(), representation2.cpu())


if __name__ == "__main__":
    main()
