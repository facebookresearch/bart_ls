import os
import re
import random
from tqdm import tqdm
import json

# books3
data_path = '/fsx/xwhan/data/pretrain_corpus/yt_subs/yt_subs.jsonl'
save_dir = '/fsx/xwhan/data/pretrain_corpus/yt_subs/raw'


lines = [json.loads(l) for l in open(data_path).readlines()]

# print(open(files[0]).read())

random.shuffle(lines)

sep = int(len(lines) * 0.95)
with open(save_dir + '/train.txt', 'w') as g:
    for l in tqdm(lines[:sep]):
        text = l['text']
        text = re.sub(r'\n\s*', '\n', text) # remove multiple newlines
        g.write(text.strip() + '\n\n')


with open(save_dir + '/valid.txt', 'w') as g:
    for l in tqdm(lines[sep:]):
        text = l['text']
        text = re.sub(r'\n\s*', '\n', text) # remove multiple newlines
        g.write(text.strip() + '\n\n')
