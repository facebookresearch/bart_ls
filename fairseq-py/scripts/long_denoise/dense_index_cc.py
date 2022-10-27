
from tracemalloc import start
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import ujson as json
import submitit
from pathlib import Path
import numpy as np
import math

import linecache as lc
from subprocess import check_output
from prefetch_generator import background

def wc(filename):
    return int(check_output(["wc", "-l", filename]).split()[0])

def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda()
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_cuda(value)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)

def encode(model, encoded):
    inputs = move_to_cuda(dict(encoded))
    outputs = model(**inputs)

    def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
    return embeddings

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def get_shared_folder() -> Path:
    return Path('/checkpoints/xwhan/indexing_cc')

@background()
def readlines_to_batch(file, start, end, bsz, tokenizer):
    for batch_start in range(start, end, bsz):
        batch_end = min(end, batch_start + bsz)
        lines = [json.loads(lc.getline(file, line_no + 1)) for line_no in range(batch_start, batch_end)]
        docs = [item['contents'] for item in lines]
        docs_encoded = tokenizer(docs, padding=True, truncation=True, return_tensors='pt')
        ids = [item['id'] for item in lines]
        yield docs_encoded, ids

def encoding(shard_id, sub_shard, num_subshards=10):

    assert sub_shard < num_subshards

    bsz = 1024
    
    # 
    corpus_path = f'/data/home/xwhan/data/long_c4/jsonl_files/shard{shard_id}/documents.jsonl'
    print('Counting docs...')
    num_docs = wc(corpus_path)
    print(f"{num_docs} docs in the file")

    shard_size = math.ceil(num_docs / num_subshards)
    start_idx = sub_shard * shard_size
    end_idx = min(start_idx + shard_size, num_docs)
    print(f"Processing lines {start_idx}-{end_idx}")

    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
    model = AutoModel.from_pretrained('facebook/contriever-msmarco')

    all_embeddings = []
    all_doc_ids = []
    with torch.no_grad():
        model.cuda().half()
        for batch_docs, batch_ids in tqdm(readlines_to_batch(corpus_path, start_idx, end_idx, bsz, tokenizer), total = math.ceil(shard_size / bsz)):
            batch_embeddings = encode(model, batch_docs)
            all_embeddings.append(batch_embeddings.cpu().numpy())
            all_doc_ids.extend(batch_ids)
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    with open(f'/data/home/xwhan/data/long_c4/jsonl_files/shard{shard_id}/embeddings_{sub_shard}.npy', 'wb') as f:
        np.save(f, all_embeddings)
    
    json.dump(all_doc_ids, open(f'/data/home/xwhan/data/long_c4/jsonl_files/shard{shard_id}/doc_ids_{sub_shard}.json', 'w'))

    print(len(all_doc_ids))
    print(all_embeddings.shape)

    return shard_id

def main():

    executor = submitit.AutoExecutor(folder=get_shared_folder() / "%j")
    executor.update_parameters(
        mem_gb=None,
        gpus_per_node=1,
        tasks_per_node=1,
        cpus_per_task=10,
        nodes=1,
        timeout_min=4320,
        slurm_partition="a100",
        slurm_job_name=f"indexing"
    )

    jobs = []
    # for shard in range(10):
    #     for sub_shard in range(10):
    job = executor.submit(encoding, 9, 9)
    jobs.append(job)
    
    all_results = []
    for job in jobs:
        all_results.extend(job.task(0).result())

if __name__ == "__main__":
    main()