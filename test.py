from family_feud_evaluator.evaluation import evaluate, family_feud, fast_money, EvalResult
from family_feud_evaluator.scoring import longest_common_subsequence_score, longest_common_substring_score, \
    wn_similarity
from family_feud_evaluator.data_processing import load_data_from_jsonl
from functools import partial
from collections import defaultdict
from typing import *

question_data = load_data_from_jsonl(
    '/home/rajarshi/Dropbox/research/family_feud_project/gpt2_inputs/11-06-first_20.jsonl')

family_feud_wn_sim = partial(family_feud, answer_score_func=wn_similarity)
fast_money_wn_sim = partial(fast_money, answer_score_func=wn_similarity)
fast_money_substring_sim = partial(fast_money, answer_score_func=longest_common_substring_score)


def read_gpt2_answers(file_name):
    q_c = 0
    answer_dict = defaultdict(list)
    prev_q = None
    for l_c, line in enumerate(open(file_name)):
        line = line.strip()
        try:
            q, a = line.split("\t")
            if prev_q != q:
                # new question:
                q_c += 1
            a = a[:-1].strip()
            if len(a) == 0:
                prev_q = q
                continue
            if a not in answer_dict["q" + str(q_c)]:
                answer_dict["q" + str(q_c)].append(a)
            prev_q = q
        except ValueError:
            print("answer not present for {}".format(line))
    return answer_dict

def print_scores(eval_result:Dict[str, EvalResult]) -> None:
    num_questions = len(eval_result)
    for qid in eval_result.keys():
        try:
            print("qid: {}, score: {}".format(qid, eval_result[qid].score))
        except KeyError:
            continue

    print("=====")



ans_dict = read_gpt2_answers("/home/rajarshi/Dropbox/research/family_feud_project/gpt2_inputs/first_20_gpt2_output.txt")
# temp = {}
# temp["q7"] = ans_dict["q7"]
# ans_dict = temp
# print(evaluate(family_feud, question_data,
#                answers_dict=ans_dict))
#
# print(evaluate(family_feud_wn_sim, question_data,
#                answers_dict=ans_dict))

print("=====fast money=======")
# print_scores(evaluate(fast_money, question_data,
#                answers_dict=ans_dict))

# print(evaluate(fast_money_substring_sim, question_data,
#                answers_dict=ans_dict))

print_scores(evaluate(fast_money_wn_sim, question_data,
               answers_dict=ans_dict))

# import pdb
# pdb.set_trace()

