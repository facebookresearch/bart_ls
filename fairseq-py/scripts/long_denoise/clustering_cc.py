import faiss
import numpy as np
from collections import defaultdict
import json
from pathlib import Path
import submitit
import hickle as hkl 
import os
import pickle
import random

# import sys, os, lucene, threading, time, csv, argparse
from datetime import datetime
from tqdm import tqdm
import json

def get_shared_folder(job_name) -> Path:
    return Path(f'/checkpoints/xwhan/{job_name}')
    
def wc(filename):
    return int(check_output(["wc", "-l", filename]).split()[0])

def sample_embedding(embedding_path, sample_ratio=0.02):
    embeddings = np.load(embedding_path)
    num_embeddings = embeddings.shape[0]
    print(embeddings.shape)
    embeddings = embeddings[np.random.choice(num_embeddings, int(sample_ratio * num_embeddings), replace=False), :]
    print(embeddings.shape)
    return embeddings.astype(np.float32)

def train_kmeans(executor, shard_dir='/data/home/xwhan/data/long_c4/jsonl_files', sample_ratio=0.05):
    """
    1. uniform sample vectors for k-means
    2. save the index for cluster assignment
    """

    print('loading embeddings with slurm')
    jobs = []
    for shard in range(10):
        for sub_shard in (range(10)):
            embedding_path = f'{shard_dir}/shard{shard}/embeddings_{sub_shard}.npy'
            job = executor.submit(sample_embedding, embedding_path, sample_ratio)
            jobs.append(job)

    sample_embeddings = []
    for job in jobs:
        sample_embeddings.append(job.task(0).result())

    sample_embeddings = np.concatenate(sample_embeddings, axis=0)
    print(f'Total sampled vectors: {sample_embeddings.shape[0]}')
    index = faiss.IndexFlatIP(768)
    kmeans = faiss.Clustering(768, 256)
    kmeans.niter = 40
    kmeans.verbose = True
    kmeans.max_points_per_centroid = int(2000000 * sample_ratio) # total 340M vectors

    kmeans.train(sample_embeddings, index)
    # saving the index
    faiss.write_index(index, f'{shard_dir}/cluster_index.bin')

def assign_to_cluster(shard_id=0, shard_dir='/data/home/xwhan/data/long_c4/jsonl_files', cluster_dir='/data/home/xwhan/data/long_c4/clusters'):
    """
    In each shard of the whole corpus, assign documents & embeddings to each cluster
    """
    
    index = faiss.read_index(f'{shard_dir}/cluster_index.bin')

    cluster2docs = defaultdict(list)
    cluster2embeddings = defaultdict(list)
    for sub_shard in tqdm(range(10)):
        batch_doc_ids = json.load(open(f'{shard_dir}/shard{shard_id}/doc_ids_{sub_shard}.json'))
        batch_embeddings = np.load(f'{shard_dir}/shard{shard_id}/embeddings_{sub_shard}.npy')
        D, I = index.search(batch_embeddings.astype(np.float32), 1)
        
        for doc_id, cluster_id, doc_embed in zip(batch_doc_ids, I.squeeze(1).tolist(), batch_embeddings):
            cluster2docs[cluster_id].append(doc_id)
            cluster2embeddings[cluster_id].append(np.expand_dims(doc_embed, 0))

    json.dump(cluster2docs, open(f'{shard_dir}/shard{shard_id}/cluster_assignment.json', 'w'))

    print('Saving embedddings and doc idx into each cluster')
    for cluster, doc_ids in tqdm(cluster2docs.items()):
        global_doc_ids = [f'{shard_id}_{idx}' for idx in doc_ids]
        embeddings = np.concatenate(cluster2embeddings[cluster], axis=0)

        data = {
            'doc_ids': global_doc_ids,
            'embeddings': embeddings
        }
        if not os.path.exists(f'{cluster_dir}/cluster_{cluster}'):
            os.mkdir(f'{cluster_dir}/cluster_{cluster}')
        hkl.dump(data, f'{cluster_dir}/cluster_{cluster}/shard_{shard_id}.hkl')

    return shard_id

def add_docs_to_cluster(shard_id=0, c4_raw='/fsx/xwhan/data/pretrain_corpus/c4/raw'):

    cluster2docs = json.load(open(f'/data/home/xwhan/data/long_c4/jsonl_files/shard{shard_id}/cluster_assignment.json'))
    file =  f'train_{shard_id}.txt'
    raw_docs = {}
    window = []
    doc_cnt = 0
    for line in tqdm(open(os.path.join(c4_raw, file)).readlines()):
        if len(line.strip()) != 0:
            window.append(line.strip())
        else:
            raw_docs[doc_cnt] = '\n'.join(window)
            doc_cnt += 1
            window = []

    for cluster_id, doc_ids in cluster2docs.items():
        save_dir = f'/data/home/xwhan/data/long_c4/clusters/cluster_{cluster_id}/documents_{shard_id}.jsonl'
        with open(save_dir, 'w') as g:
            for doc_id in tqdm(doc_ids):
                global_id = f'{shard_id}_{doc_id}'
                g.write(json.dumps({
                    'id': global_id,
                    'raw': raw_docs[doc_id]
                }) + '\n')
    
    return shard_id

def distribute_docs(executor, shard_dir='/data/home/xwhan/data/long_c4/jsonl_files'):
    jobs = []
    for shard_id in range(1, 10):
        # job = executor.submit(assign_to_cluster, shard_id, shard_dir)

        job = executor.submit(add_docs_to_cluster, shard_id, '/fsx/xwhan/data/pretrain_corpus/c4/raw')
        jobs.append(job)
    all_results = []
    for job in jobs:
        all_results.append(job.task(0).result())
    print(all_results)

def search_in_cluster(cluster_id, parent_dir='/data/home/xwhan/data/long_c4/clusters'):
    """
    dense search in each clusters, find knns for each documents
    """

    cluster_dir = f'{parent_dir}/cluster_{cluster_id}'

    print(cluster_dir)

    doc_ids = []
    embeddings = []
    for shard_id in tqdm(range(10)):
        shard = hkl.load(f'{cluster_dir}/shard_{shard_id}.hkl')
        embeddings.append(shard['embeddings'].astype(np.float32))
        doc_ids.extend(shard['doc_ids'])
    embeddings = np.concatenate(embeddings, axis=0)
    
    print(embeddings.shape)

    print("Moving to gpu index")
    search_bsz = 256
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    flat_config.useFloat16 = True
    index = faiss.GpuIndexFlatIP(res, 768, flat_config)
    index.add(embeddings)

    # search_results = {}
    topk_table = []
    for b_start in tqdm(range(0, len(doc_ids), search_bsz)):
        batch_embeds = embeddings[b_start:b_start+search_bsz, :]
        batch_docs = doc_ids[b_start:b_start+search_bsz]
        _, I = index.search(batch_embeds, 2048) # 
        for doc_id, topk in zip(batch_docs, I):
            # topk = [doc_ids[_] for _ in topk if doc_ids[_] != doc_id]
            # search_results[doc_id] = topk
            topk_table.append(topk)
    
    # pickle.dump(search_results, open(f'{cluster_dir}/knn_100.pkl', 'wb'))
    import h5py
    hf = h5py.File(f'{cluster_dir}/knn_2k.h5', 'w')
    hf.create_dataset('knn', data=np.array(topk_table, dtype=np.uint32))
    hf.close()
    return cluster_id

def knn_docs(executor, cluster_dir='/data/home/xwhan/data/long_c4/clusters'):
    num_clusters = 256
    knn_doc = []

    jobs = []
    for cluster_id in range(2, num_clusters):
        # job = executor.submit(search_in_cluster, cluster_id, cluster_dir)
        job = executor.submit(group_in_cluster, cluster_id, cluster_dir)
        jobs.append(job)

    for job in jobs:
        knn_doc.append(job.task(0).result())
    print(knn_doc)
    

def assemble_long_in_cluster(cluster_id, cluster_dir='/fsx/xwhan/data/pretrain_corpus/c4/long/clusters'):
    """
    Make
    """

    import h5py
    import re
    print('Load knn results...')
    # knn_map = pickle.load(open(f'{cluster_dir}/cluster_{cluster_id}/knn_100.pkl', 'rb'))
    hf = h5py.File(f'{cluster_dir}/cluster_{cluster_id}/knn_2k.h5', 'r')
    knn = hf['knn']

    print(f'Load all documents in cluster {cluster_id}...')
    id2doc = {}
    doc_ids = []
    for shard in tqdm(range(10)):
        documents = [json.loads(l) for l in open(f'{cluster_dir}/cluster_{cluster_id}/documents_{shard}.jsonl').readlines()]
        shard = hkl.load(f'{cluster_dir}/cluster_{cluster_id}/shard_{shard}.hkl')
        doc_ids.extend(shard['doc_ids'])
        for doc in documents:
            id2doc[doc['id']] = doc['raw'].strip()

    # print(f"Start assembling long docs...")
    # seen_docs = defaultdict(int)
    # long_docs = [] # each line is a list of docs
    # long_docs_ids = [] 
    # doc_lens = []
    # for doc_id in knn_map.keys():
    #     if doc_id in seen_docs:
    #         continue
    #     seen_docs[doc_id] += 1

    #     text = id2doc[doc_id]
    #     doc_len = len(text.split())

    #     if doc_len > 10000:
    #         long_docs.append([text.strip()])
    #         long_docs_ids.append([doc_id])
    #     else:
    #         knn_docs = knn_map[doc_id]
    #         doc_list = [id2doc[doc_id].strip()]
    #         doc_id_list = [doc_id]
    #         for knn_id in knn_docs:
    #             # if seen_docs[knn_id] >= 3: # max repeition
    #             #     continue
    #             seen_docs[knn_id] += 1
    #             knn_doc = id2doc[knn_id]
    #             doc_list.append(knn_doc.strip())
    #             doc_id_list.append(knn_id)
    #             doc_len += len(knn_doc.split())
    #             if doc_len > 16000:
    #                 break
    #         long_docs.append(doc_list)
    #         long_docs_ids.append(doc_id_list)
    #     doc_lens.append(doc_len)

    orig_doc_and_len = [(doc_id, len(id2doc[doc_id].split())) for doc_id in id2doc.keys()]
    orig_doc_and_len.sort(key=lambda x:x[1])

    knn_map = {}
    for doc_id, topk in tqdm(zip(doc_ids, knn)):
        topk_ = [doc_ids[_] for _ in topk if doc_ids[_] != doc_id]
        knn_map[doc_id] = topk_

    print(f"Start assembling long docs by recursive retrieval...")
    visited = defaultdict(int)
    long_docs = [] # each line is a list of docs
    long_docs_ids = [] 
    doc_lens = []

    for doc_id, _ in tqdm(orig_doc_and_len):
        if doc_id in visited:
            continue
        visited[doc_id] += 1
        text = id2doc[doc_id]
        doc_len = len(text.split())

        if doc_len > 10000:
            long_docs.append([text.strip()])
            long_docs_ids.append([doc_id])
        else:
            # recursively retrieval
            linked_doc_ids = set([doc_id]) # current traversal
            linked_docs = [text.strip()]
            query_id = doc_id
            while True:
                knn_ids = knn_map[query_id]
                first_unseen = None
                for item in knn_ids:
                    if item not in linked_doc_ids and visited[item] < 2:
                        first_unseen = item
                        break
                if first_unseen is None:
                    break
            
                next_hop_id = first_unseen
                next_hop_doc = id2doc[next_hop_id]
                linked_docs.append(next_hop_doc)
                linked_doc_ids.add(next_hop_id)
                doc_len += len(next_hop_doc.split())
                visited[next_hop_id] += 1

                if doc_len > 14000 or len(linked_docs) >= 150:
                    break

                query_id = next_hop_id

            long_docs.append(linked_docs)
            long_docs_ids.append(list(linked_doc_ids))

        doc_lens.append(doc_len)

    freqs = np.array(list(visited.values()))
    doc_lens = np.array(doc_lens)

    # failed cases
    assembled, failed = [], []
    for doc_list, doc_len in zip(long_docs, doc_lens):
        if doc_len < 14000:
            failed.append((doc_list, doc_len))
        else:
            assembled.append(doc_list)

    random.shuffle(failed)
    long = []
    curr_len = 0
    for doc_list, doc_len in failed:
        curr_len += doc_len
        long.extend(doc_list)
        if curr_len >= 14000:
            assembled.append(long)
            long = []
            curr_len = 0
    if curr_len != 0:
        assembled.append(long)

    print(len(assembled))

    save_path = f'/fsx/xwhan/data/pretrain_corpus/assembled_c4_top2k/cluster_{cluster_id}.txt'
    with open(save_path, 'w') as f:
        for idx, docs in enumerate(assembled):
            random.shuffle(docs)
            docs = [re.sub(r'\n+', '\n', d).strip() for d in docs]
            docs = [d for d in docs if len(d) > 0]
            if len(docs) > 0:
                f.write('\n'.join(docs).strip() + '\n\n')
            else:
                print(assembled[idx])
    return cluster_id

def group_in_cluster(cluster_id, parent_dir='/data/home/xwhan/data/long_c4/clusters'):
    """
    Using clustering instead of retrieval to assemble documents
    """
    cluster_dir = f'{parent_dir}/cluster_{cluster_id}'
    print(cluster_dir)

    doc_ids = []
    embeddings = []
    for shard_id in tqdm(range(10)):
        shard = hkl.load(f'{cluster_dir}/shard_{shard_id}.hkl')
        embeddings.append(shard['embeddings'].astype(np.float32))
    
        doc_ids.extend(shard['doc_ids'])
    embeddings = np.concatenate(embeddings, axis=0)
    
    num_docs = embeddings.shape[0]

    num_clusters = num_docs // 100

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    flat_config.useFloat16 = True
    index = faiss.GpuIndexFlatIP(res, 768, flat_config)
    kmeans = faiss.Clustering(768, num_clusters)
    kmeans.niter = 20
    kmeans.verbose = True
    kmeans.min_points_per_centroid = 50
    kmeans.max_points_per_centroid = 400
    kmeans.train(embeddings, index)
    
    cluster2docs = defaultdict(list)
    batch_size = 5000
    for batch_start in range(0, num_docs, batch_size):

        batch_doc_ids = doc_ids[batch_start:batch_start + batch_size]
        batch_embeddings = embeddings[batch_start:batch_start + batch_size,:]

        D, I = index.search(batch_embeddings.astype(np.float32), 1)
        for doc_id, cluster_id, _ in zip(batch_doc_ids, I.squeeze(1).tolist(), batch_embeddings):
            cluster2docs[cluster_id].append(doc_id)
    
    # # save the documents groups
    # group_sizes = []
    # for k, v in cluster2docs.items():
    #     group_sizes.append(len(v))
    pickle.dump(cluster2docs, open(f'{cluster_dir}/groups.pkl', 'wb'))

    return cluster_id

def assemble_long_by_group(cluster_id, cluster_dir='/data/home/xwhan/data/long_c4/clusters'):
    """
    Assemble long documents by randomly sample documents from clusters
    """
    print('Load group results...')
    groups = pickle.load(open(f'{cluster_dir}/cluster_{cluster_id}/groups.pkl', 'rb'))

    print(f'Load all documents in cluster {cluster_id}...')
    id2doc = {}
    for shard in range(10):
        documents = [json.loads(l) for l in open(f'{cluster_dir}/cluster_{cluster_id}/documents_{shard}.jsonl').readlines()]
        for doc in documents:
            id2doc[doc['id']] = doc['raw']

    print(f"Start assembling long docs...")
    long_docs = [] # each line is a list of docs
    doc_lens = []

    for c_id, doc_list in tqdm(groups.items()):
        random.shuffle(doc_list)
        curr_doc_len = 0
        curr_docs = []
        for doc_id in doc_list:
            doc = id2doc[doc_id]
            curr_docs.append(doc)
            curr_doc_len += len(doc.split())
            if curr_doc_len > 14000:
                long_docs.append(curr_docs)
                doc_lens.append(curr_doc_len)
                curr_docs = []
                curr_doc_len = 0
        if len(curr_docs) != 0:
            long_docs.append(curr_docs)
            doc_lens.append(curr_doc_len)  

    doc_lens = np.array(doc_lens)
    breakpoint()
    # save_path = f'/data/home/xwhan/data/long_c4/small_clusters/texts/cluster_{cluster_id}.txt'
    # with open(save_path, 'w') as f:
    #     for docs in long_docs:
    #         random.shuffle(docs)
    #         f.write('\n'.join(docs) + '\n\n')


def gather_knn(executor):
    cluster_dir='/data/home/xwhan/data/long_c4/clusters'
    jobs = []
    for cluster_id in range(256):
        # job = executor.submit(assemble_long_by_group, cluster_id, cluster_dir)
        job = executor.submit(assemble_long_in_cluster, cluster_id, cluster_dir)
        jobs.append(job)
    all_results = []
    for job in jobs:
        all_results.append(job.task(0).result())
    print(all_results)


def reshard_docs(executor, datadir='/data/home/xwhan/data/long_c4/assembled_top2k/texts'):
    """
    INPUTS: Long documents in each clusters

    Shuffle documents in each cluster and distribute to shards
    """
    import random
    import math

    def split_cluster(cluster_id, datadir, save_dir):

        shards = 10

        file = f'{datadir}/cluster_{cluster_id}.txt'

        print(f'Reading docs {file}')
        docs = []
        window = []
        for line in tqdm(open(os.path.join(file)).readlines()):
            if len(line.strip()) != 0:
                window.append(line.strip())
            else:
                docs.append('\n'.join(window))
                window = []

        random.shuffle(docs)
        shard_sz = math.ceil(len(docs) / shards)
    
        for shard in range(shards):

            if not os.path.exists(save_dir + f'/{shard}'):
                os.mkdir(save_dir + f'/{shard}')

            start_idx = shard * shard_sz
            shard_docs = docs[start_idx:start_idx + shard_sz]

            print(f"Saving {start_idx}-{start_idx + shard_sz}")
            with open(f'{save_dir}/{shard}/c_{cluster_id}.txt', 'w') as g:
                for doc in shard_docs:
                    g.write(doc.strip() + '\n\n')
        
        return cluster_id

    save_dir = '/fsx/xwhan/data/pretrain_corpus/assembled_c4_top2k/shards'

    jobs = []
    for cluster_id in range(256):
        job = executor.submit(split_cluster, cluster_id, datadir, save_dir)
        jobs.append(job)

    all_results = []
    for job in jobs:
        all_results.append(job.task(0).result())

    print(all_results)


if __name__ == "__main__":
    job_name = 'assemble_from_top2k'

    executor = submitit.AutoExecutor(folder=get_shared_folder(job_name) / "%j")
    executor.update_parameters(
        mem_gb=None,
        gpus_per_node=1,
        tasks_per_node=1,
        cpus_per_task=10,
        nodes=1,
        timeout_min=4320,
        slurm_partition="a100",
        slurm_job_name=job_name,
    )
    # train_kmeans(executor) # load and sample document embeddings and train a cluster model
    # distribute_docs(executor) # create cluster dir and save documents in each cluster
    # knn_docs(executor) # knn search in each cluster
    # gather_knn(executor) # assemble long documents in each cluster and save raw text
    reshard_docs(executor)

    # group_in_cluster(2)
    # search_in_cluster(190)
    # assemble_long_in_cluster(45)
