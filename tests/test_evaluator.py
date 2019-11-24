import pytest
from family_feud_evaluator import *
from family_feud_evaluator.evaluation import *
from functools import partial
from pathlib import Path

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from nltk.metrics.distance import jaro_winkler_similarity

eval_methods = {
    'family_feud': family_feud,
    'fast_money': fast_money,
    'family_feud_2_incorrect': partial(family_feud, max_incorrect=2),
    'family_feud_5_incorrect': partial(family_feud, max_incorrect=5),
    'set_intersection': set_intersection,
    'soft_jaro_winkler_set_intersection': partial(set_intersection, answer_cluster_scoring_func=jaro_winkler_similarity),
    'hard_jaro_winkler_set_intersection': partial(set_intersection,
                                                  answer_cluster_scoring_func=jaro_winkler_similarity,
                                                  score_matrix_transformation=lambda z: np.round(z)
                                                  ),
    'hard_lcsubstring_set_int': partial(set_intersection,
                                answer_cluster_scoring_func=longest_common_substring_score,
                                score_matrix_transformation=lambda z: np.round(z)
                                ),
    'hard_lcsubseq_set_int': partial(set_intersection,
                                answer_cluster_scoring_func=longest_common_subsequence_score,
                                score_matrix_transformation=lambda z: np.round(z)
                                ),
}

answer_set_10_60_30 = {"10": 10, "60": 60, "30": 30}
answer_set_Apple10_Bananna60_Carrot30 = {"apple": 10, "bananna": 60, "carrot": 30}
answer_set_12_27_48 = {'12': 12, '27': 27, '48': 48}

testdata = (
    ('exact', ['60', '30', '10'], answer_set_10_60_30, {'family_feud': 1.0, 'fast_money': 1.0, 'set_intersection': 1.0, 'soft_jaro_winkler_set_intersection': 1.0}),
    ('exact_less_than_100', ['48', '27', '12'], answer_set_12_27_48, {'family_feud': 1.0, 'fast_money': 1.0, 'set_intersection': 1.0}),
    ('no_match', ['a', 'b', 'c', 'd', 'e'], answer_set_10_60_30, {'family_feud': 0.0, 'fast_money': 0.0, 'set_intersection': 0.0}),
    ('scale_to_max', ['30', 'a', 'b'], answer_set_10_60_30, {'family_feud': 0.3, 'fast_money': 0.5, 'set_intersection': 1/3}),
    ('wrong_order', ['10', '30', '60'], answer_set_10_60_30, {'family_feud': 1.0, 'fast_money': 1/6, 'set_intersection': 1.0}),
    ('no_double_counting', ['60', '60', '60'], answer_set_10_60_30, {'family_feud': 0.6, 'fast_money': 1, 'set_intersection': 1/3}),
    ('three_wrong', ['30', 'X', 'X', '10', 'X', '60'], answer_set_10_60_30, {'family_feud': 0.4, 'fast_money': 0.5, 'set_intersection': 1.0}),
    ('three_wrong_right_away', ['X', 'X', 'X', '10', 'X', '60'], answer_set_10_60_30, {'family_feud': 0.0, 'fast_money': 0.0, 'family_feud_5_incorrect': 0.7, 'set_intersection': 2/3}),
    ('two_wrong', ['30', 'X', '10', 'X', '60'], answer_set_10_60_30, {'family_feud': 1.0, 'family_feud_2_incorrect': 0.4, 'set_intersection': 1.0}),
    ('many_repeats_should_not_penalize', ['30', '30', '30', '30', '30','30','60'], answer_set_10_60_30, {'family_feud': 0.9, 'family_feud_2_incorrect': 0.9, 'family_feud_5_incorrect': 0.9, 'set_intersection': 2/3}),
    ('sloppy_input_answers', ['an Apple', 'X', 'the Banannnaa'], answer_set_Apple10_Bananna60_Carrot30, {'family_feud': 0.0, 'set_intersection': 0.0, 'hard_jaro_winkler_set_intersection': 2/3, 'hard_lcsubstring_set_int': 1/3, 'hard_lcsubseq_set_int': 2/3}),
)

testdata_param_dict = {k + '][' + e[0]: (eval_methods[k], e[1], e[2], v) for e in testdata for k, v in e[3].items()}


@pytest.mark.parametrize(
    "eval_method, pred_answers, true_answers, expected",
    list(testdata_param_dict.values()),
    ids=list(testdata_param_dict.keys()),
)
def test_parametrized(eval_method, pred_answers, true_answers, expected):
    assert eval_method(pred_answers, true_answers) == expected


@pytest.fixture()
def data_path():
    mod_path = Path(__file__).parent
    return mod_path / "data_stub.jsonl"

def test_load_data(data_path):
    load_data_from_jsonl(data_path)

@pytest.fixture()
def question_data(data_path):
    return load_data_from_jsonl(data_path)


def test_access_data(question_data):
    for i in range(5):
        question_data[f'q{i}']


def test_evaluate_single_question(question_data):
    assert evaluate(family_feud, question_data, answers_dict={'q0':["umbrella", "hat", "towel"]}) == {'q0':38/99}

@pytest.fixture()
def answers_5():
    return {
        'q0': ['umbrella', 'sunscreen', 'towel', 'sun glasses'],
        'q1': ['bed', 'shower', 'bathroom'],
        'q2': ['baby crying'],
        'q3': ['10'], # These aren't great... we might want to filter out answers which are purely numbers.
        'q4': ['40'],
    }


def test_evaluate_multiple_questions(answers_5, question_data):
    assert evaluate(set_intersection, question_data, answers_dict=answers_5) == {'q0': 2/6, 'q1': 2/7, 'q2': 0, 'q3': 1/7, 'q4':1/5}


def test_readme_example(question_data):
    soft_lcsubsequence_set_int = partial(
        general_eval,
        answer_cluster_scoring_func=longest_common_subsequence_score,
        assign_cluster_scores = False, # This is what makes it a set, it turns off the cluster counts
    )

    assert evaluate(soft_lcsubsequence_set_int, question_data,
                    answers_dict={'q0': ['umbrella', 'hat', 'sun glasses']}) == {'q0': 0.3896103896103896}
