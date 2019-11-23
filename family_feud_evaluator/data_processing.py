from pathlib import Path
from typing import *
import json
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

# stop_words = set(stopwords.words('english'))

def load_data_from_jsonl(data_path:Union[Path,str]):
    question_data = dict()
    with open(data_path) as data:
        for q in data:
            q_json = json.loads(q)
            question_data[q_json['questionid']] = q_json
    return question_data

# def filter_questions:
#     nltk.download('punkt') # needed for tokenization


def default_string_preprocessing(pred_answer: str, length_limit:int = 50) -> str:
    return pred_answer.lower()[:length_limit]


