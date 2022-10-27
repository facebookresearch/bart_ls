from datasets import load_dataset
from tqdm import tqdm
import submitit
from pathlib import Path
import math
from fairseq.data.encoders.gpt2_bpe import get_encoder
from multiprocessing import Pool

def get_shared_folder() -> Path:
    return Path('/checkpoints/xwhan/jobs')


class MultiprocessingEncoder(object):
    def __init__(self):
        self.encoder_json = '/datasets01/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin/121219/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin/encoder.json'
        self.vocab_bpe = '/datasets01/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin/121219/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin/vocab.bpe'
        self.keep_empty = True

    def initializer(self):
        global bpe
        bpe = get_encoder(self.encoder_json, self.vocab_bpe)

    def encode(self, line):
        global bpe
        ids = bpe.encode(line)
        return list(map(str, ids))

    def decode(self, tokens):
        global bpe
        return bpe.decode(tokens)

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = line.strip()
            if len(line) == 0 and not self.keep_empty:
                return ["EMPTY", None]
            tokens = self.encode(line)
            enc_lines.append(" ".join(tokens))
        return ["PASS", enc_lines]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]

executor = submitit.AutoExecutor(folder=get_shared_folder() / "%j")
executor.update_parameters(
    mem_gb=None,
    gpus_per_node=1,
    tasks_per_node=1,
    cpus_per_task=10,
    nodes=1,
    timeout_min=4320,
    slurm_partition="a100",
)

def read_text(shard_id, shards=8):
    print("Loading data...")
    dataset = load_dataset('c4', 'en', split='train')
    shard_size = math.ceil(len(dataset) / shards)
    start_idx = shard_id * shard_size
    
    save_path = f'/fsx/xwhan/data/pretrain_corpus/c4/train_{shard_id}.txt'

    with open(save_path, 'w') as g:
        for idx in tqdm(range(start_idx, start_idx + shard_size)):
            line = dataset[idx]
            g.write(line['text'].strip() + "\n\n")
    return shard_id


def read_long_text(shard_id):
    print("Loading data...")

    bpe_shards = f"/fsx/xwhan/data/pretrain_corpus/c4/bpe/train_{shard_id}.bpe"

    # bpe_shards = '/fsx/xwhan/data/pretrain_corpus/c4/bpe/valid.bpe'

    long_docs = []
    curr_doc, curr_len = [], 0
    doc_cnt = 0
    for line in tqdm(open(bpe_shards).readlines()):
        if len(line.strip()) == 0:
            doc_cnt += 1
            if curr_len > 4000:
                long_docs.append("\n".join(curr_doc))
            curr_doc, curr_len = [], 0
        else:
            curr_doc.append(line.strip())
            curr_len += len(line.strip().split())
    

    print(f'Found {len(long_docs)} long docs over {doc_cnt} docs')
    save_dir = '/fsx/xwhan/data/pretrain_corpus/c4/over_4k_bpe'

    # with open(save_dir + f'/train_{shard_id}.bpe', 'w') as g:
    with open(save_dir + f'/valid.bpe', 'w') as g:
        for long_doc in long_docs:
            g.write(long_doc.strip() + '\n\n')

    return shard_id
    

# read_long_text(0)

shards = 10
jobs = []
for shard_id in range(1, shards):
    job = executor.submit(read_long_text, shard_id)
    jobs.append(job)

results = [job.task(0).result() for job in jobs]
print(f"Jobs results: {results}")