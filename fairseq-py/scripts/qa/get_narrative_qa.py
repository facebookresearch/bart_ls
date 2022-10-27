from email.policy import default
from xml.sax import default_parser_list
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict
import os
import re
import json

dataset = load_dataset("narrativeqa", split='test')
id2qa = defaultdict(dict)

# check if one document has multiple questions
q_cnt = 0
for item in dataset:
    doc_id = item['document']['id']
    id2qa[doc_id][item['question']['text'].strip()] = [a['text'] for a in item['answers']]
    q_cnt += 1

scrolls_datasets = ["narrative_qa"]
data = [load_dataset("tau/scrolls", dataset) for dataset in scrolls_datasets]
narrative_test = data[0]['test']


print(len(narrative_test))

save_dir = '/fsx/xwhan/data/narrativeqa'
split = 'test'

question_set = set()

with open(os.path.join(save_dir, f'{split}.query'), 'w') as q_f, open(os.path.join(save_dir, f'{split}.source'), 'w') as s_f, open(os.path.join(save_dir, f'{split}.uid'), 'w') as t_f, open(os.path.join(save_dir, f'{split}.target'), 'w') as a_f:
 
    for item in tqdm(narrative_test):
        doc_id = item['id'].split('_')[0]

        question = item['input'].split('\n\n')[0]
        context = item['input'][len(question):].strip()

        context = re.sub(r'\n\s*', ' ', context)
        # breakpoint()
        answers = id2qa[doc_id][question.strip()]

        assert '\n' not in question, question
        # raw = id2raw[doc_id]
        # item['answers'] = answers

        q_f.write(question.strip() + '\n')
        s_f.write(context + '\n')
        t_f.write(item['id'] + '\n')
        a_f.write(json.dumps(answers) + '\n')

breakpoint()