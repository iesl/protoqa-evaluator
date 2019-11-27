from pathlib import Path
from typing import *
import json
from collections import Counter
import hashlib

try:
    import pandas as pd
    import xlrd
    CROWDSOURCE_CONVERSION = True
except:
    CROWDSOURCE_CONVERSION = False


def load_data_from_jsonl(data_path: Union[Path,str]) -> Dict:
    question_data = dict()
    with open(data_path) as data:
        for q in data:
            q_json = json.loads(q)
            question_data[q_json['questionid']] = q_json
    return question_data


def load_data_from_excel(data_path: Union[Path, str], next_idx: int = 0) -> Dict:
    if CROWDSOURCE_CONVERSION:
        data_path = Path(data_path)
        with open(data_path, mode='rb') as f:
            data_hash = hashlib.md5(f.read()).hexdigest()
            sheets = pd.read_excel(f, sheet_name = None)
        question_data = dict()
        for sheet_idx, (sheet_name, sheet) in enumerate(sheets.items()):
            q_dict = dict()
            q_dict['raw-original-answers'] = Counter(sheet['answer'].dropna())
            q_dict['raw-answers-cleaned'] = Counter(sheet['fixed_spelling'].dropna())
            clusters = sheet.loc[sheet['fixed_spelling'].notna() & (sheet['Combined'] != '?'), ['fixed_spelling', 'Combined']] \
                .groupby('Combined')['fixed_spelling'].agg(count=len, frozenset=frozenset)
            q_dict['answers-cleaned'] = {row['frozenset']: row['count'] for _, row in clusters.iterrows()}
            questionid = f'q{next_idx + sheet_idx}'
            question_data[questionid] = {
                'question': sheet['question'][0],
                'normalized-question': sheet['question'][0].lower(),
                'source': data_path.name,
                'source-md5': data_hash,
                'sourceid': sheet_name,
                'questionid': questionid,
                **q_dict,
            }
        return question_data
    else:
        raise Exception('Was not able to import pandas and xlrd, which is required for conversion.')


def default_string_preprocessing(pred_answer: str, length_limit: int = 50) -> str:
    return pred_answer.lower()[:length_limit]


