from dataclasses import dataclass
from pathlib import Path
from typing import *
import json


def family_feud(pred_answers: List[str], true_answers: Dict[str, int], *args, max_incorrect:int = 3, **kwargs):
    pred_answers = pred_answers.copy()
    true_answers = true_answers.copy()
    score = 0
    max_score = sum(true_answers.values())
    incorrect = 0
    for i, answer in enumerate(pred_answers):
        try:
            score += true_answers.pop(answer)
        except KeyError:
            incorrect += 1
            if incorrect >= max_incorrect:
                break
    score /= max_score
    return score


def fast_money(pred_answers, true_answers):
    pred_answers = pred_answers.copy()
    true_answers = true_answers.copy()
    score = true_answers.get(pred_answers[0], 0)
    score /= max(true_answers.values())
    return score


class Evaluator:
    def __init__(self, *data_paths):
        self.data_paths = []
        for d in data_paths:
            d = Path(d)
            if d.is_dir():
                self.data_paths += d.glob('*.jsonl')
            else:
                self.data_paths.append(d)
        self._questions = dict()
        self._load_data_from_path()

    def _load_data_from_path(self):
        for path in self.data_paths:
            with open(path) as data:
                for q in data:
                    q_json = json.loads(q)
                    # If we were sure this will run on Python >= 3.6 we could
                    # rely on the fact that dictionaries are ordered, but for
                    # simplicity and compatibility we will make it a list.
                    # answer_list = list(q_json['answers-cleaned'].keys())
                    # answer_list = sorted(answer_list, key=lambda x: q_json['answers-cleaned'][x])
                    # q_json['answer-list'] = answer_list
                    self._questions[q_json['questionid']] = q_json

    def __call__(self, methods: Optional[Dict[str, Callable]] = None, **qs_to_eval: List[str]) -> Dict:
        """
        Returns scores according to various evaluation metrics.
        :param kwargs: dict {q#: [answer list]}
        :return: dict of scores {q#: {metric_name: score} for each q#}
        """
        if methods is None:
            methods = {'family feud': family_feud, 'fast money': fast_money}
        scores = dict()
        for qid, pred_answers in qs_to_eval.items():
            true_q = self._questions[qid]
            true_answers = true_q['answers-cleaned'].copy()
            score = dict()
            for mname, mfunc in methods.items():
                score[mname] = mfunc(pred_answers, true_answers)
            scores[qid] = score
        return scores
