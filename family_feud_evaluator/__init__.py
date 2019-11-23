from __future__ import annotations
from .scoring import *
from .data_processing import load_data_from_jsonl
from .evaluation import family_feud, fast_money
from typing import *

if TYPE_CHECKING:
    from pathlib import Path


class Evaluator:
    def __init__(self, data_path:Union[Path,str]):
        self._questions = load_data_from_jsonl(data_path)

    def __call__(self, answer_assignment: Optional[Dict[str, Callable]] = None,
                 methods: Optional[Dict[str, Callable]] = None, **qs_to_eval: List[str]) -> Dict:
        """
        Returns scores according to various evaluation metrics.
        :param answer_assignment: dict {assignment_name: assignment_func}
        :param : dict {assignment_name: assignment_func}
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
