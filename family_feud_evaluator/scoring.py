import numpy as np
from scipy.optimize import linear_sum_assignment
from difflib import SequenceMatcher
from itertools import product
from typing import *
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

EN_STOPWORDS = set(stopwords.words('english'))


##########################################################################
# Functions which take in a score matrix and return the actual score
##########################################################################
def get_optimal_score(score_matrix: np.ndarray) -> float:
    cost_matrix = - score_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return score_matrix[row_ind, col_ind].sum(), row_ind, col_ind

#################################################################
# Functions which take a single pred_answer and true_answer,
# and return a score.
##################################################################
def exact_match(pred_answer: str, true_answer: str) -> float:
    return float(pred_answer == true_answer)


def longest_common_substring_score(pred_answer: str, true_answer: str) -> float:
    sm = SequenceMatcher(None, pred_answer, true_answer)
    match = sm.find_longest_match(0, len(pred_answer), 0, len(true_answer))
    return match.size / max(len(pred_answer), len(true_answer))


def longest_common_subsequence_score(pred_answer: str, true_answer: str) -> float:
    sm = SequenceMatcher(None, pred_answer, true_answer)
    lcsubseq_size = sum([block.size for block in sm.get_matching_blocks()])
    return lcsubseq_size / max(len(pred_answer), len(true_answer))


def wn_similarity(pred_answer: str, true_answer: str, remove_stopwords: bool = True, sim_fn: Callable = wn.wup_similarity,
                  score_reduction_fn: Callable = lambda z: get_optimal_score(z)[0]) -> float:
    """
    Computes WN based similarity between two strings. I am using the Wu-Palmer similarity by default, although
    I couldnt find a clear citation where one would work better than the other. This stack exchange answer is the
    best resource -- https://linguistics.stackexchange.com/questions/9084/what-do-wordnetsimilarity-scores-mean

    :param pred_answer:
    :param true_answer:
    :param remove_stopwords:
    :param sim_fn:
    :param score_reduction_fn:
    :return:
    """
    def _wn_sim(tok1: str, tok2: str, sim_fn: Callable = sim_fn) -> float:
        """
        calculates the max similairty between two tokens using the similarity function by checking with all their synsets
        :param tok1:
        :param tok2:
        :return:
        """
        tok1_synsets = wn.synsets(tok1)
        tok2_synsets = wn.synsets(tok2)
        if len(tok1_synsets) == 0 or len(tok2_synsets) == 0:
            return -1.0
        sim_mat = np.ones((len(tok1_synsets), len(tok2_synsets)), dtype="float32") * -1.0
        for i, syn_t1 in enumerate(tok1_synsets):
            for j, syn_t2 in enumerate(tok2_synsets):
                sim = sim_fn(syn_t1, syn_t2)
                if sim is not None:
                    sim_mat[i, j] = sim

        return np.max(sim_mat)

    # if exact match, then return 1. This check is to take care of situations where
    # a wordnet synset is not present of the word.
    if exact_match(pred_answer, true_answer) == 1.0:
        return 1.0
    pred_ans_tokens = word_tokenize(pred_answer)
    true_answer_tokens = word_tokenize(true_answer)
    # remove stop words if necessary
    if remove_stopwords:
        pred_ans_tokens = [tok for tok in pred_ans_tokens if tok not in EN_STOPWORDS]
        true_answer_tokens = [tok for tok in true_answer_tokens if tok not in EN_STOPWORDS]

    score_mat = np.empty((len(pred_ans_tokens), len(true_answer_tokens)))

    for i, pred_tok in enumerate(pred_ans_tokens):
        for j, true_tok in enumerate(true_answer_tokens):
            score_mat[i, j] = _wn_sim(pred_tok, true_tok)

    score_token_wise_match = score_reduction_fn(score_mat) / len(pred_ans_tokens)
    # now match the whole strings and see if there is a similarity
    score_full_match = _wn_sim("_".join(pred_ans_tokens), "_".join(true_answer_tokens))
    score = max(score_token_wise_match, score_full_match)
    return score


##########################################################################
# Functions which take a list of pred_answers and true_answers,
# and return a score matrix (pred_answers x true_answers)
##########################################################################
def pred_true_pairwise_scores(pred_answers: List[str], true_answers: Union[Dict[str, int], Dict[frozenset, int]],
                              answer_score_func: Callable = exact_match,
                              answer_score_reduction_func: Callable = max) -> np.ndarray:
    pred_answers = pred_answers.copy()
    true_answers = true_answers.copy()
    score_matrix = np.empty((len(pred_answers), len(true_answers)))
    for (pred_i, pred_a), (true_j, true_a) in product(enumerate(pred_answers), enumerate(true_answers)):
        if isinstance(true_a, frozenset):
            scores = [answer_score_func(pred_a, true_a_i) for true_a_i in true_a]
            score = answer_score_reduction_func(scores)
        else:
            score = answer_score_func(pred_a, true_a)
        score_matrix[pred_i, true_j] = score
    return score_matrix


##########################################################################
# Functions which take in a score matrix and return an augmented
# score matrix
##########################################################################
def scale_score_matrix_by_cluster_scores(score_matrix: np.ndarray,
                                         cluster_scores: List[int]) -> np.ndarray:
    return score_matrix * np.array(cluster_scores)[None]


def limit_total_wrong(score_matrix: np.ndarray, k: int) -> np.ndarray:
    answer_scores = score_matrix.max(axis=1)
    incorrect = 0
    for i, a in enumerate(answer_scores):
        if a == 0:
            incorrect += 1
            if incorrect >= k:
                return score_matrix[:i + 1]
    return score_matrix
