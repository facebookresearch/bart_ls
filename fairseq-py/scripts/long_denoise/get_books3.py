import glob
import os
import re
import random
import math
from tqdm import tqdm

# books3
data_dir = '/fsx/xwhan/data/pretrain_corpus/books3/the-eye.eu/public/Books/Bibliotik'
save_dir = '/fsx/xwhan/data/pretrain_corpus/books3/raw'

# gutenberg
# data_dir = '/fsx/xwhan/data/pretrain_corpus/gutenberg-dammit-files'
# save_dir = '/fsx/xwhan/data/pretrain_corpus/gutenberg/raw'

os.chdir(data_dir)

files = glob.glob("**/*.txt", recursive=True)

# print(open(files[0]).read())

random.shuffle(files)
shards = 10
shard_size = math.ceil(len(files) / shards)
for shard_id in range(shards):
    
    save_path = os.path.join(save_dir, f'{shard_id}.txt')
    start_idx = shard_id * shard_size

    with open(save_path, 'w') as g:
        for file in tqdm(files[start_idx: start_idx + shard_size]):
            content = open(file).readlines()
            content = list(filter(lambda x: not re.match(r'^\s*$', x), content)) # remove empty lines
            g.write("".join(content).strip() + '\n\n')
