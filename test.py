from family_feud_evaluator.evaluation import evaluate, family_feud, fast_money
from family_feud_evaluator.scoring import longest_common_subsequence_score, longest_common_substring_score, wn_similarity
from family_feud_evaluator.data_processing import load_data_from_jsonl
from functools import partial

question_data = load_data_from_jsonl('tests/crowdsource_data_stub.jsonl')

family_feud_wn_sim = partial(family_feud, answer_score_func=wn_similarity)
fast_money_wn_sim = partial(fast_money, answer_score_func=wn_similarity)


print(evaluate(family_feud, question_data,
               answers_dict={'q0': ['asteroids and stuff', 'comets', 'sun'], 'q1': ["potato", "squash", "tomato", "you know, veggies"]}))

print(evaluate(family_feud_wn_sim, question_data,
               answers_dict={'q0': ['asteroids and stuff', 'comets', 'sun'], 'q1': ["potato", "squash", "tomato", "you know, veggies"]}))


print(evaluate(fast_money, question_data,
               answers_dict={'q0': ['halley\'s comet'], 'q1': ["potato"]}))

print(evaluate(fast_money_wn_sim, question_data,
               answers_dict={'q0': ['halley\'s comet'], 'q1': ["potato"]}))