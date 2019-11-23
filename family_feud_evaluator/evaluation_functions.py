from .scoring import *
from functools import partial
from typing import *


def general_eval(pred_answers, true_answers,
                 max_pred_answers: Optional[int] = None,
                 max_incorrect: Optional[int] = None,
                 answer_cluster_scoring_func: Callable = exact_match,
                 assign_cluster_scores: bool = True,
                 calc_oracle_score: bool = True,
                 ) -> float:
    if max_pred_answers is not None:
        pred_answers = pred_answers[:max_pred_answers]
    score_matrix = pred_true_pairwise_scores(pred_answers, true_answers, answer_cluster_scoring_func)
    if max_incorrect is not None:
        score_matrix = limit_total_wrong(score_matrix, max_incorrect)
    if assign_cluster_scores:
        score_matrix *= np.array(list(true_answers.values()))[None]
    score = get_optimal_score(score_matrix)
    if calc_oracle_score:
        oracle_answers = sorted(list(true_answers.keys()), key=lambda z: true_answers[z], reverse=True)
        oracle_score = general_eval(pred_answers=oracle_answers, true_answers=true_answers,
                                     max_pred_answers=max_pred_answers, max_incorrect=max_incorrect,
                                     answer_cluster_scoring_func=answer_cluster_scoring_func,
                                     assign_cluster_scores=assign_cluster_scores,
                                     calc_oracle_score=False)
        score /= oracle_score
    return score

fast_money = partial(general_eval, max_pred_answers = 1)

family_feud = partial(general_eval, max_incorrect = 3)


# Direct implementations of some of the simpler algorithms,
# without the functional structure of the general setting.
# Useful for testing, in case something in the more general setting goes wrong.
def naive_family_feud(pred_answers: List[str], true_answers: Dict[str, int], *args, max_incorrect:int = 3, **kwargs):
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


def naive_fast_money(pred_answers, true_answers):
    pred_answers = pred_answers.copy()
    true_answers = true_answers.copy()
    score = true_answers.get(pred_answers[0], 0)
    score /= max(true_answers.values())
    return score


