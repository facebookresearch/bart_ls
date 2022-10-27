import zstd
import json
import os
from tqdm import tqdm

data_path = "/fsx/xwhan/data/pretrain_corpus/dialogue/opensub"


all_docs = []

for file in tqdm(os.listdir(data_path + "/out")):
    with open(os.path.join(data_path, "out", file), 'rb') as f:
        cdata = f.read()
    data = zstd.decompress(cdata)
    data = json.loads(data)
    all_docs.extend(data)

# using the last 1000 as valid set
with open(data_path + '/train.txt', 'w') as f:
    for doc in all_docs[:-1000]:
        f.write("\n".join(doc))
        f.write("\n\n")

with open(data_path + '/valid.txt', 'w') as f:
    for doc in all_docs[-1000:]:
        f.write("\n".join(doc))
        f.write("\n\n")
