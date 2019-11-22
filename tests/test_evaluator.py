import pytest
from family_feud_evaluator import *
from functools import partial

eval_methods = {
    'family_feud': family_feud,
    'fast_money': fast_money,
    'family_feud_2_incorrect': partial(family_feud, max_incorrect=2),
}

answer_set_10_60_30 = {"10": 10, "60": 60, "30": 30}
answer_set_12_27_48 = {'12': 12, '27': 27, '48': 48}

def test_fast_money_between_match(answer_set_60_30_10):
    assert fast_money(["one", "30", "three"], answer_set_60_30_10) == 0
    # Note, this is 0.5 because the max points would have been 60, and we got 30.

testdata = (
    ('exact', ['60', '30', '10'], answer_set_10_60_30, {'family_feud': 1.0, 'fast_money': 1.0}),
    ('exact_less_than_100', ['48', '27', '12'], answer_set_12_27_48, {'family_feud': 1.0, 'fast_money': 1.0}),
    ('no_match', ['a', 'b', 'c', 'd', 'e'], answer_set_10_60_30, {'family_feud': 0.0, 'fast_money': 0.0}),
    ('scale_to_max', ['30', 'a', 'b'], answer_set_10_60_30, {'family_feud': 0.3, 'fast_money': 0.5}),
    ('wrong_order', ['10', '30', '60'], answer_set_10_60_30, {'family_feud': 1.0, 'fast_money': 1/6}),
    ('no_double_counting', ['60', '60', '60'], answer_set_10_60_30, {'family_feud': 0.6, 'fast_money': 1}),
    ('three_wrong', ['30', 'X', 'X', '10', 'X', '60'], answer_set_10_60_30, {'family_feud': 0.4, 'fast_money': 0.5}),
    ('two_wrong', ['30', 'X', '10', 'X', '60'], answer_set_10_60_30, {'family_feud': 1.0, 'family_feud_2_incorrect': 0.4}),
)

testdata_param_dict = {k + '][' + e[0]: (eval_methods[k], e[1], e[2], v) for e in testdata for k, v in e[3].items()}


@pytest.fixture()
def data_path():
    return "data_stub.jsonl"

@pytest.fixture()
def evaluator(data_path):
    return Evaluator(data_path)

@pytest.fixture()
def answer_set_60_30_10():
    # Intentionally done in a mixed up order - shouldn't matter.
    return {"10": 10, "60": 60, "30": 30}

def test_load_data(data_path):
    Evaluator(data_path)

def test_access_data(evaluator):
    for i in range(5):
        evaluator._questions[f'q{i}']

@pytest.mark.parametrize(
    "eval_method, pred_answers, true_answers, expected",
    list(testdata_param_dict.values()),
    ids=list(testdata_param_dict.keys()),
)
def test_parametrized(eval_method, pred_answers, true_answers, expected):
    assert eval_method(pred_answers, true_answers) == expected

def test_actual_no_match(evaluator):
    assert evaluator(q0=["something else"]) == {'q0':{'family feud': 0.0, 'fast money': 0.0}}

def test_actual_no_double_counting(evaluator):
    assert evaluator(q0=["umbrella", "umbrella", "umbrella", "umbrella"]) == {'q0': {'family feud': 38/99, 'fast money': 1}}