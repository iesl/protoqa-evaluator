#    Copyright 2022 The ProtoQA Evaluator Authors.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from functools import partial

import numpy as np
from protoqa_evaluator.evaluation import general_eval
from protoqa_evaluator.scoring import wordnet_score

__all__ = [
    "max_answers",
    "max_incorrect",
    "exact_match_all_eval_funcs",
    "wordnet_all_eval_funcs",
    "all_eval_funcs",
    "fast_money",
    "family_feud",
    "set_intersection",
    "hard_set_intersection",
]

max_answers = {
    f"Max Answers - {k}": partial(general_eval, max_pred_answers=k)
    for k in [None, 1, 3, 5, 10]
}
max_incorrect = {
    f"Max Incorrect - {k}": partial(general_eval, max_incorrect=k)
    for k in [None, 1, 3, 5]
}
exact_match_all_eval_funcs = {**max_answers, **max_incorrect}
wordnet_all_eval_funcs = {
    k: partial(v, score_func=wordnet_score, score_matrix_transformation=np.round)
    for k, v in exact_match_all_eval_funcs.items()
}
all_eval_funcs = {
    "exact_match": exact_match_all_eval_funcs,
    "wordnet": wordnet_all_eval_funcs,
}

fast_money = partial(general_eval, max_pred_answers=1)
family_feud = partial(general_eval, max_incorrect=3)
set_intersection = partial(general_eval, assign_cluster_scores=False)
hard_set_intersection = partial(set_intersection, score_matrix_transformation=np.round)
