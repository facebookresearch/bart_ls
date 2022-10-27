import json
from tqdm import tqdm

data_dir = "/fsx/xwhan/data/pretrain_corpus/dialogue/mediasum"
data_path = "/fsx/xwhan/data/pretrain_corpus/dialogue/news_dialogue.json"

all_data = json.load(open(data_path))


# 463596 dialogue in total, using 3000 for validation

raw_data = []
for item in tqdm(all_data):
    utts = item["utt"]
    speakers = item["speaker"]
    assert len(utts) == len(speakers)
    doc = [s + ": " + u.strip() for u,s in zip(utts, speakers)]
    raw_data.append(doc)

with open(data_dir + '/train.txt', 'w') as f:
    for doc in raw_data[:-3000]:
        f.write("\n".join(doc))
        f.write("\n\n")

with open(data_dir + '/valid.txt', 'w') as f:
    for doc in raw_data[-3000:]:
        f.write("\n".join(doc))
        f.write("\n\n")

