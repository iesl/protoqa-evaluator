from typing import *
import numpy as np
from itertools import product
from scipy.optimize import linear_sum_assignment
from difflib import SequenceMatcher


#################################################################
# Functions which take a single pred_answer and true_answer,
# and return a score.
##################################################################
def exact_match(pred_answer: str, true_answer: str) -> float:
    return float(pred_answer == true_answer)


def longest_common_substring_ratio(pred_answer: str, true_answer: str) -> float:
    sm = SequenceMatcher(None, pred_answer.lower(), true_answer)
    match = sm.find_longest_match(0, len(pred_answer), 0, len(true_answer))
    return match.size / max(len(pred_answer), len(true_answer))

##########################################################################
# Functions which take a list of pred_answers and true_answers,
# and return a score matrix (pred_answers x true_answers)
##########################################################################
def pred_true_pairwise_scores(pred_answers: List[str], true_answers: Dict[str, int],
                              answer_score_func: Callable = exact_match) -> np.ndarray:
    pred_answers = pred_answers.copy()
    true_answers = true_answers.copy()
    score_matrix = np.empty((len(pred_answers), len(true_answers)))
    for (pred_i, pred_a), (true_j, true_a) in product(enumerate(pred_answers), enumerate(true_answers)):
        score_matrix[pred_i, true_j] = answer_score_func(pred_a, true_a)
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
                return score_matrix[:i+1]
    return score_matrix


##########################################################################
# Functions which take in a score matrix and return the actual score
##########################################################################
def get_optimal_score(score_matrix:np.ndarray) -> float:
    cost_matrix = - score_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return score_matrix[row_ind, col_ind].sum()
