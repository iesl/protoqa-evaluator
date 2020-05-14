import argparse
from functools import partial
from typing import *

import numpy as np
from tabulate import tabulate

from family_feud_evaluator.data_processing import load_data_from_jsonl, load_predictions
from family_feud_evaluator.evaluation import (
    evaluate,
    family_feud,
    fast_money,
    set_intersection,
    family_feud_5_incorrect,
)
from family_feud_evaluator.scoring import (
    longest_common_subsequence_score,
    longest_common_substring_score,
    wordnet_score,
    exact_match,
)

family_feud_wn_sim = partial(family_feud, score_func=wordnet_score)
fast_money_wn_sim = partial(fast_money, score_func=wordnet_score)
fast_money_substring_sim = partial(
    fast_money, score_func=longest_common_substring_score
)

EVAL_METHODS = [
    ("Fast Money", fast_money),
    ("Family Feud", family_feud),
    ("Set Intersection", set_intersection),
    ("Family Feud (5 incorrect)", family_feud_5_incorrect),
]
SIM_FUNCS = [
    ("Exact Match", exact_match),
    ("Longest Substr", longest_common_substring_score),
    ("Longest Subseq", longest_common_subsequence_score),
    ("WordNet", wordnet_score),
]
HARD_SOFT = [("Hard", np.round), ("Soft", None)]


def calc_scores(predictions: Dict[str, List[str]], question_data: Dict) -> float:

    all_rows = []
    for eval_method in EVAL_METHODS:
        rows = [eval_method[0]]
        for sim_func in SIM_FUNCS:
            for hs in HARD_SOFT:
                print("Eval method: {}".format(eval_method[0]))
                print("Similarity Function: {}".format(sim_func[0]))
                print("Hard/Soft: {}".format(hs[0]))
                method = partial(
                    eval_method[1],
                    score_func=sim_func[1],
                    score_matrix_transformation=hs[1],
                )
                scores = evaluate(
                    method, question_data=question_data, answers_dict=predictions
                )
                avg_score = np.mean([s.score for _, s in scores.items()])
                # for _, s in scores.items():
                #     print(s.answer_assignment)
                # print(avg_score)
                rows.append(avg_score)
        all_rows.append(rows)
    header = ["Eval method"]
    for s in SIM_FUNCS:
        for hs in HARD_SOFT:
            header.append(s[0] + " ({})".format(hs[0]))
    print(tabulate(all_rows, headers=header))


def main(args):

    question_data = load_data_from_jsonl(args.ground_truth_annotation_file)
    predictions = load_predictions(args.prediction_file)
    calc_scores(predictions, question_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect subgraphs around entities")
    parser.add_argument("--prediction_file", type=str, required=True)
    parser.add_argument("--ground_truth_annotation_file", type=str, required=True)
    args = parser.parse_args()
    main(args)
