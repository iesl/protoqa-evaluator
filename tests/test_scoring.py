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

from typing import *
import pytest

from protoqa_evaluator.scoring import *


class AnswerScoreExample(NamedTuple):
    pred_answer: str
    true_answer: str
    funcs_to_eval: Dict[Callable, float]


# shorthand
ASE = AnswerScoreExample

test_answer_score_examples = {
    "wn_same_synset": ASE(
        "jump",
        "startle",
        {
            wordnet_score: 1,
            exact_match: 0,
            longest_common_substring_score: 0,
            longest_common_subsequence_score: 0,
        },
    ),
    "wn_oov": ASE("oov", "oov", {wordnet_score: 1}),
    "wn_partial_oov": ASE("oov plant", "oov flora", {wordnet_score: 1}),
    "wn_should_not_match": ASE(
        "ear muffs", "ear wax", {wordnet_score: 0.5, wordnet_wup_score: 2 / 3}
    ),
    "wn_gum": ASE("gum", "chewing gum", {wordnet_score: 1}),
}


def convert_to_param_dict(test_data):
    return {
        func.__name__
        + "]["
        + test_name: (func, ase.pred_answer, ase.true_answer, expected)
        for test_name, ase in test_data.items()
        for func, expected in ase.funcs_to_eval.items()
    }


test_answer_score_examples_param = convert_to_param_dict(test_answer_score_examples)


@pytest.mark.parametrize(
    "func, pred_answer, true_answer, expected",
    list(test_answer_score_examples_param.values()),
    ids=list(test_answer_score_examples_param.keys()),
)
def test_answer_score_example(func, pred_answer, true_answer, expected):
    assert func(pred_answer, true_answer) == expected
