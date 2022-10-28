# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from tqdm import tqdm
import json
import os
from nltk import word_tokenize
from pathlib import Path

# import stanza
# nlp = stanza.Pipeline(lang='en', processors={'tokenize': 'spacy'})

# tokneize a sent
def tokenize(sent):
    tokens = ' '.join(word_tokenize(sent.lower()))
    return tokens

    # doc = nlp(sent)
    # return ' '.join(token.text for sentence in doc.sentences for token in sentence.tokens)

def clean_data(text):
    text = text.replace('{ vocalsound } ', '')
    text = text.replace('{ disfmarker } ', '')
    text = text.replace('a_m_i_', 'ami')
    text = text.replace('l_c_d_', 'lcd')
    text = text.replace('p_m_s', 'pms')
    text = text.replace('t_v_', 'tv')
    text = text.replace('{ pause } ', '')
    text = text.replace('{ nonvocalsound } ', '')
    text = text.replace('{ gap } ', '')
    return text


def simplify(data_dir, save_dir, oracle=False):

    if oracle:
        save_dir = Path(save_dir + '-gold')
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'val', 'test']:
        
        data_path = os.path.join(data_dir, f'{split}.jsonl')
        
        data = []
        with open(data_path) as f:
            for line in f:
                data.append(json.loads(line))

        queries, sources, targets = [], [], []
        targets_raw = []
        for i in range(len(data)):
            src = []
            for k in range(len(data[i]['meeting_transcripts'])):
                cur_turn = data[i]['meeting_transcripts'][k]['speaker'].lower() + ': '
                cur_turn = cur_turn + tokenize(data[i]['meeting_transcripts'][k]['content'])
                src.append(cur_turn)
            src = ' '.join(src)

            for j in range(len(data[i]['general_query_list'])):
                query = tokenize(data[i]['general_query_list'][j]['query'])
                queries.append(clean_data(query))
                sources.append(clean_data(src))
                targets_raw.append(data[i]['general_query_list'][j]['answer'])
                targets.append(tokenize(data[i]['general_query_list'][j]['answer']))

            for j in range(len(data[i]['specific_query_list'])):
                query = tokenize(data[i]['specific_query_list'][j]['query'])
                target = tokenize(data[i]['specific_query_list'][j]['answer'])
                
                if oracle:
                    relevant = []
                    for span in data[i]['specific_query_list'][j]['relevant_text_span']:
                        assert len(span) == 2
                        st, ed = int(span[0]), int(span[1])
                        for k in range(st, ed + 1):
                            cur_turn = data[i]['meeting_transcripts'][k]['speaker'].lower() + ': '
                            cur_turn = cur_turn + tokenize(data[i]['meeting_transcripts'][k]['content'])
                            relevant.append(cur_turn)
                    sources.append(clean_data(" ".join(relevant)))
                else:
                    sources.append(clean_data(src))
                queries.append(clean_data(query))
                targets_raw.append(data[i]['specific_query_list'][j]['answer'])
                targets.append(target)
                                
        with open(save_dir / f'{split}.source', 'w') as g1, \
            open(save_dir /  f'{split}.target', 'w') as g2,  \
            open(save_dir /  f'{split}.query', 'w') as g3, \
            open(save_dir / f'{split}.target_raw', 'w') as g4:

            for q, s, t, t_ in zip(queries, sources, targets, targets_raw):
                g1.write(s.strip() + '\n')
                g2.write(t.strip() + '\n')
                g3.write(q.strip() + '\n')
                g4.write(t_.strip() + '\n')
            

if __name__ == '__main__':

    data_dir = '/fsx/xwhan/data/QMSum/data/ALL/jsonl'
    save_dir = '/fsx/xwhan/data/QMSum/data/raw'

    simplify(data_dir, save_dir, oracle=False)