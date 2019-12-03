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


def default_string_preprocessing(pred_answer: str, length_limit: int = 50) -> str:
    return pred_answer.lower()[:length_limit]


def load_data_from_jsonl(data_path: Union[Path,str]) -> Dict:
    question_data = dict()
    with open(data_path) as data:
        for q in data:
            q_json = json.loads(q)
            if isinstance(q_json['answers-cleaned'], list):
                q_json['answers-cleaned'] = {frozenset(ans_cluster['answers']): ans_cluster['count'] for ans_cluster in q_json['answers-cleaned']}
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
            q_dict['raw-original-answers'] = Counter(sheet['answer'].dropna().astype(str))
            q_dict['raw-answers-cleaned'] = Counter(sheet['fixed_spelling'].dropna().astype(str))
            clusters = sheet.loc[sheet['fixed_spelling'].notna() & (sheet['Combined'] != '?'), ['fixed_spelling', 'Combined']] \
                .astype(str).groupby('Combined')['fixed_spelling'].agg(count=len, frozenset=frozenset)
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


def save_question_cluster_data_to_jsonl(data_path: Union[Path, str], q_dict: Dict) -> None:
    with open(data_path, 'w') as output_file:
        for q in q_dict.values():
            q = q.copy()
            q['answers-cleaned'] = [{'count': count, 'answers': list(answers)} for answers, count in q['answers-cleaned'].items()]
            json.dump(q, output_file)
            output_file.write('\n')


