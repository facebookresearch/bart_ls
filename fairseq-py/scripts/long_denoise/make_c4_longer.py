import submitit
import os
import ujson as json
from tqdm import tqdm
from pathlib import Path
import math
import random
from nltk.util import ngrams
from collections import Counter
import six
from multiprocessing import Pool
import sys

# def _score_ngrams(target_ngrams, prediction_ngrams):
#     """Compute n-gram overlap scores

#     each ngram is counted once as in Pegasus paper
#     """
#     target_ngrams = set(target_ngrams.keys())
#     prediction_ngrams = set(prediction_ngrams.keys())
#     intersection_ngrams_count = len(target_ngrams.intersection(prediction_ngrams))

#     target_ngrams_count = len(target_ngrams)
#     prediction_ngrams_count = len(prediction_ngrams)

#     precision = intersection_ngrams_count / max(prediction_ngrams_count, 1)
#     recall = intersection_ngrams_count / max(target_ngrams_count, 1)

#     if precision + recall > 0:
#         return 2 * precision * recall / (precision + recall)
#     else:
#         return 0.0

def _score_ngrams(target_ngrams, prediction_ngrams):
    """Compute n-gram based rouge scores.
    Args:
        target_ngrams: A Counter object mapping each ngram to number of
        occurrences for the target text.
        prediction_ngrams: A Counter object mapping each ngram to number of
        occurrences for the prediction text.
    Returns:
        A Score object containing computed scores.
    """

    intersection_ngrams_count = 0
    for ngram in six.iterkeys(target_ngrams):
        intersection_ngrams_count += min(target_ngrams[ngram],
                                        prediction_ngrams[ngram])
    target_ngrams_count = sum(target_ngrams.values())
    prediction_ngrams_count = sum(prediction_ngrams.values())

    precision = intersection_ngrams_count / max(prediction_ngrams_count, 1)
    recall = intersection_ngrams_count / max(target_ngrams_count, 1)

    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    else:
        return 0.0

def ngram_counter(doc, ns=[1,2]):
    doc = doc.lower()
    results = []
    for n in ns:
        for ngram in ngrams(doc.split(), n):
            results.append(" ".join(ngram))
    return Counter(results)

def downsample(doc, max_len=400):
    """
    reduce the document lengths to maximum 400 tokens, 
    such that the texts are within the lengths limit of dense retrievers
    """
    all_ngrams_counter = ngram_counter(doc)
    sents = doc.split('\n')
    sent_and_scores = []
    for sent_id, sent in enumerate(sents):
        sent = sent.strip()
        sent_counter = ngram_counter(sent)
        rest_counter = all_ngrams_counter - sent_counter
        score = _score_ngrams(rest_counter, sent_counter)
        sent_and_scores.append((sent, score, sent_id))

    sent_and_scores.sort(reverse=True, key=lambda x:x[1])

    doc_small = []
    doc_len = 0
    for sent, score, sent_id in sent_and_scores:
        doc_small.append((sent, sent_id))
        doc_len += len(sent.split())
        if doc_len > max_len:
            break
    
    doc_small.sort(key=lambda x:x[1])

    # keep the original sentence order
    return ' '.join([_[0] for _ in doc_small])

def reduce_doc(doc_object):
    contents = downsample(doc_object['raw'])
    return {
        'id': doc_object['id'],
        'contents': contents
    }

def formatting(shard_id):
    """
    Shorten documents based on primary sentences if necessary
    """

    print(f'shard no. {shard_id}')
    c4_raw = '/fsx/xwhan/data/pretrain_corpus/c4/raw'
    
    docs = []
    file =  f'train_{shard_id}.txt'
    window = []
    doc_cnt = 0
    for line in tqdm(open(os.path.join(c4_raw, file)).readlines()):
        if len(line.strip()) != 0:
            window.append(line.strip())
        else:
            docs.append({
                'id': doc_cnt,
                'raw': '\n'.join(window)
            })
            doc_cnt += 1
            window = []

    save_dir = '/data/home/xwhan/data/long_c4/jsonl_files'
    print(f'{len(docs)} documents collected......')
    if not os.path.exists(save_dir + f'/shard{shard_id}'):
        os.mkdir(save_dir + f'/shard{shard_id}')


    pool = Pool(32)
    reduced_docs = pool.imap(reduce_doc, docs, chunksize=500)

    with open(save_dir + f'/shard{shard_id}/documents.jsonl', 'w') as g:
        for i, doc in enumerate(reduced_docs, start=1):
            g.write(json.dumps(doc) + '\n')
            if i % 50000 == 0:
                print("processed {} docs".format(i), file=sys.stderr)

    return shard_id


# def assemble_long(shard_id, sub_shard_id, sub_shards=10, batch_size=1000, threads=24):

#     def reduce_query(query, max_len = 48):
#         query_len = len(query.split())
#         if query_len < max_len:
#             return query
#         else:
#             start = random.randint(0, query_len - max_len)
#             query_tokens = query.split()
#             return " ".join(query_tokens[start:start + max_len])


#     print(f'batch size: {batch_size} and threads: {threads}')

#     print('Loading index...')
#     index_path = f'/fsx/xwhan/data/pretrain_corpus/c4/indexes/shard{shard_id}'
#     searcher = LuceneSearcher(index_path)

#     print('Loading documents...')
#     corpus_path = f'/data/home/xwhan/data/long_c4/jsonl_files/shard{shard_id}/documents.jsonl'
#     id2docs = {str(json.loads(line)['id']):json.loads(line)['contents'] for line in tqdm(open(corpus_path).readlines())}

#     num_docs = searcher.num_docs
#     shard_size = math.ceil(num_docs / sub_shards)
#     start_idx = sub_shard_id * shard_size

#     knn_dict = {}
#     batch = []

#     print('Searching')
#     for doc_id in tqdm(range(start_idx, start_idx + shard_size)):
#         doc_id = str(doc_id)

#         condensed_doc = reduce_query(id2docs[doc_id])

#         batch.append({
#             'id': doc_id,
#             'contents': condensed_doc
#         })
#         if len(batch) == batch_size:
#             batch_queries = [condensed_doc for _ in batch]
#             batch_qids = [_['id'] for _ in batch] 
#             batch_results = searcher.batch_search(batch_queries, batch_qids, k=6, threads=threads) # top 10 related docs
#             for qid in batch_qids:
#                 results = batch_results[qid]
#                 results_ids = [_.docid for _ in results if _.docid != qid]
#                 knn_dict[qid] = results_ids
#             batch = []


def get_shared_folder() -> Path:
    return Path('/checkpoints/xwhan/jobs')


if __name__ == '__main__':
    shard_id = int(sys.argv[1])
    formatting(shard_id)
    # assemble_long(shard_id, 0)
    
    # executor = submitit.AutoExecutor(folder=get_shared_folder() / "%j")
    # executor.update_parameters(
    #     mem_gb=None,
    #     gpus_per_node=4,
    #     tasks_per_node=1,
    #     cpus_per_task=10,
    #     nodes=1,
    #     timeout_min=4320,
    #     slurm_partition="a100",
    # )

    # jobs = []
    # for shard_id in range(6, 8):
    #     # formatting(shard_id)
    #     job = executor.submit(formatting, shard_id)
        # jobs.append(job)

    # print(f"Jobs results: {results}")


    # assemble_long('0_0', 0, sub_shards=20)



