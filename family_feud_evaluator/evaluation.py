from .data_processing import default_string_preprocessing
from .scoring import *


def evaluate(
    evaluation_func: Callable,
    question_data: Dict,
    answers_dict: Dict[str, List[str]],
    data_preprocessing: Optional[Callable] = None,
) -> Dict[str, float]:
    scores = dict()
    for qid in question_data.keys():
        pred_answers = answers_dict[qid]
        true_q = question_data[qid]
        if data_preprocessing is not None:
            true_q, pred_answers = data_preprocessing(true_q, answers_dict)
        true_answers = true_q["answers-cleaned"].copy()
        scores[qid] = evaluation_func(pred_answers, true_answers, question_string=true_q['normalized-question'])
    return scores


class EvalResult(NamedTuple):
    score: float
    score_matrix: np.ndarray
    answer_assignment: dict

    def __eq__(self, other):
        return (
            self.score == other.score
            and (self.score_matrix == other.score_matrix).all()
            and self.answer_assignment == other.answer_assignment
        )


def general_eval(
    pred_answers,
    true_answers,
    *,
    max_pred_answers: Optional[int] = None,
    max_incorrect: Optional[int] = None,
    string_preprocessing: Callable = default_string_preprocessing,
    question_string: str = "question string",
    score_func: Callable = exact_match,
    cluster_score_func: Callable = cluster_score,
    cluster_reduction_func: Callable = np.max,
    score_matrix_transformation: Optional[Callable] = None,
    assign_cluster_scores: bool = True,
    calc_oracle_score: bool = True,
) -> EvalResult:
    if max_pred_answers is not None:
        pred_answers = pred_answers[:max_pred_answers]
    pred_answers = [string_preprocessing(pred_answer) for pred_answer in pred_answers]
    score_matrix = cluster_score_func(
        pred_answers,
        true_answers,
        question_string = question_string,
        score_func=score_func,
        cluster_reduction_func=cluster_reduction_func,
    )
    if score_matrix_transformation is not None:
        score_matrix = score_matrix_transformation(score_matrix)
    if max_incorrect is not None:
        score_matrix = limit_total_wrong(score_matrix, max_incorrect)
    if assign_cluster_scores:
        score_matrix *= np.array(list(true_answers.values()))[None]
    score, row_ind, col_ind = get_optimal_score(score_matrix)
    answer_assignment = dict()
    true_answers_list = list(true_answers.keys())
    for r, c in zip(row_ind, col_ind):
        answer_assignment[pred_answers[r]] = (
            true_answers_list[c] if score_matrix[r, c] > 0 else None
        )
    if calc_oracle_score:
        oracle_answers = sorted(
            list(true_answers.keys()), key=lambda z: true_answers[z], reverse=True
        )
        if isinstance(oracle_answers[0], frozenset):
            oracle_answers = [ans for (ans, *_) in oracle_answers]
        oracle_score, *_ = general_eval(
            pred_answers=oracle_answers,
            true_answers=true_answers,
            max_pred_answers=max_pred_answers,
            max_incorrect=max_incorrect,
            string_preprocessing=string_preprocessing,
            question_string=question_string,
            score_func=score_func,
            cluster_score_func=cluster_score_func,
            cluster_reduction_func=cluster_reduction_func,
            score_matrix_transformation=score_matrix_transformation,
            assign_cluster_scores=assign_cluster_scores,
            calc_oracle_score=False,
        )
        score /= oracle_score
    return EvalResult(
        score=score, score_matrix=score_matrix, answer_assignment=answer_assignment
    )


fast_money = partial(general_eval, max_pred_answers=1, calc_oracle_score=True)

family_feud = partial(general_eval, max_incorrect=3, calc_oracle_score=True)

family_feud_2_incorrect = partial(general_eval, max_incorrect=2, calc_oracle_score=True)

family_feud_5_incorrect = partial(general_eval, max_incorrect=5, calc_oracle_score=True)

set_intersection = partial(general_eval, assign_cluster_scores=False)

hard_set_intersection = partial(set_intersection, score_matrix_transformation=np.round)

mlm_family_feud = partial(general_eval, max_incorrect=3, cluster_score_func=cluster_score_considering_whole_cluster)
maxpred1 = partial(general_eval, max_pred_answers=1, cluster_score_func=cluster_score_considering_whole_cluster)
maxpred3 = partial(general_eval, max_pred_answers=3, cluster_score_func=cluster_score_considering_whole_cluster)
maxpred5 = partial(general_eval, max_pred_answers=5, cluster_score_func=cluster_score_considering_whole_cluster)
maxpred10 = partial(general_eval, max_pred_answers=10, cluster_score_func=cluster_score_considering_whole_cluster)
maxinc1 = partial(general_eval, max_pred_answers=1, cluster_score_func=cluster_score_considering_whole_cluster)
maxinc3 = partial(general_eval, max_pred_answers=3, cluster_score_func=cluster_score_considering_whole_cluster)
maxinc5 = partial(general_eval, max_pred_answers=5, cluster_score_func=cluster_score_considering_whole_cluster)

# Direct implementations of some of the simpler algorithms,
# without the functional structure of the general setting.
# Useful for testing, in case something in the more general setting goes wrong.
def naive_family_feud(
    pred_answers: List[str],
    true_answers: Dict[str, int],
    *args,
    max_incorrect: int = 3,
    **kwargs,
) -> float:
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
