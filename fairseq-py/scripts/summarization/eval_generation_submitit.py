# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Parallel testing using submitit
"""

from tqdm import tqdm
import submitit
from pathlib import Path
import math
import os

import torch
import argparse
from tqdm import tqdm
from fairseq import checkpoint_utils
from fairseq.tasks.summarization import load_langpair_dataset
from fairseq.models.bart.hub_interface import BARTHubInterface
from fairseq.data.encoders.gpt2_bpe import GPT2BPE, GPT2BPEConfig



def get_shared_folder() -> Path:
    return Path('/checkpoints/xwhan/eval_summ_jobs')

def generate(args, shard_id, generate_args):

    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([args.model_dir])
    model = models[0]

    task.cfg.required_seq_len_multiple = 1024
    task.cfg.left_pad_source = False
    dataset_for_inference = load_langpair_dataset(args.data_dir, 
    split=args.split,
    src='src' if 'tv' not in args.data_dir else 'source',
    src_dict=task.src_dict,
    tgt='tgt' if 'tv' not in args.data_dir else 'target',
    tgt_dict=task.tgt_dict,
    combine=False,
    dataset_impl=task.cfg.dataset_impl,
    upsample_primary=task.cfg.upsample_primary,
    left_pad_source=task.cfg.left_pad_source,
    left_pad_target=task.cfg.left_pad_target,
    max_source_positions=task.cfg.max_source_positions,
    max_target_positions=task.cfg.max_target_positions,
    truncate_source=True,
    shuffle=False,
    )

    hub_interface = BARTHubInterface(cfg, task, model)
    bpe_cfg = GPT2BPEConfig()
    hub_interface.bpe = GPT2BPE(bpe_cfg)
    
    hub_interface.cfg.dataset.batch_size = args.bsz
    all_sents = [item['source'] for item in dataset_for_inference]

    shard_size = math.ceil(len(all_sents) / args.shards)
    start_idx = shard_id * shard_size

    shard_sents = all_sents[start_idx:start_idx + shard_size]

    shard_results = []
    with torch.no_grad():
        hub_interface = hub_interface.eval()
        assert torch.cuda.is_available()
        hub_interface = hub_interface.cuda().half()
        for idx in tqdm(range(0, len(shard_sents), args.bsz)):
            batch_sents = shard_sents[idx:idx+args.bsz]
            batch_hypos = hub_interface.generate(
                batch_sents,
                skip_invalid_size_inputs=False,
                **generate_args
            )
            batch_outputs = [hub_interface.decode(hypos[0]["tokens"]) for hypos in batch_hypos]
            shard_results.extend(batch_outputs)
    return shard_results

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        required=True,
        type=str,
        default="bart.large.cnn/",
        help="path containing model file and src_dict.txt",
    )

    parser.add_argument(
        "--data-dir", default="binarized data path", help="text to summarize", type=str
    )
    parser.add_argument("--save-dir", default=None, type=str)
    parser.add_argument("--bsz", default=4, help="batch size", type=int)

    parser.add_argument(
        "--n", default=None, help="how many examples to summarize", type=int
    )
    parser.add_argument("--split", default='test', type=str)
    parser.add_argument("--shards", default=8, type=int)
    parser.add_argument("--skip-eval", action='store_true', default=False)

    args = parser.parse_args()


    data_name = args.data_dir.split('/')[-1][:-len("-bin")]
    raw_data_path = args.data_dir[:-len("-bin")]

    if 'gov_report' in args.data_dir:
        generate_args = dict(beam=4,  max_len_b=740, lenpen=4.0, no_repeat_ngram_size=0, min_len=50)
    elif 'tv' in args.data_dir:
        generate_args = dict(beam=4, max_len_b=640, lenpen=5.0, no_repeat_ngram_size=3, min_len=50)
    elif 'summscreen' in args.data_dir:
        generate_args = dict(beam=4, max_len_b=350, lenpen=4.0, no_repeat_ngram_size=4, min_len=50)
    elif 'booksum' in args.data_dir:
        generate_args = dict(beam=4, max_len_b=550, lenpen=4.0, no_repeat_ngram_size=3, min_len=20)
    elif 'pubmed' in args.data_dir:
        generate_args = dict(beam=4, max_len_b=400, lenpen=4.0, no_repeat_ngram_size=3, min_len=40)
    elif 'arxiv' in args.data_dir:
        generate_args = dict(beam=4, max_len_b=300, lenpen=5.0, no_repeat_ngram_size=4, min_len=50)
    else:
        generate_args = dict(beam=4, max_len_b=256, lenpen=2.0, no_repeat_ngram_size=3, min_len=50)
    print(f'Generating parameters {generate_args}')

    executor = submitit.AutoExecutor(folder=get_shared_folder() / "%j")
    executor.update_parameters(
        mem_gb=None,
        gpus_per_node=1,
        tasks_per_node=1,
        cpus_per_task=10,
        nodes=1,
        slurm_time=120,
        timeout_min=120,
        slurm_partition="lowpri",
        slurm_job_name=f"eval_summ_{data_name}",
        slurm_exclude=os.environ.get("EXCLUDED_HOSTS", None)
    )

    jobs = []
    for shard_id in range(args.shards):
        job = executor.submit(generate, args, shard_id, generate_args)
        jobs.append(job)

    all_results = []
    for job in jobs:
        all_results.extend(job.task(0).result())
    import datetime
    # suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    # save_path = args.save_dir + '.' + suffix
    save_path = args.save_dir

    with open(save_path, 'w') as out:
        for l in all_results:
            out.write(l.strip() + '\n')

    if args.skip_eval:
        return
    
    from rouge_score import rouge_scorer
    import nltk
    all_refs = [l.strip() for l in open(f'{raw_data_path}/{args.split}.tgt').readlines()]

    scorer = rouge_scorer.RougeScorer(rouge_types=['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    rouge1 = rouge2 = rougel = 0.0
    for ref, pred in zip(all_refs, all_results):
        ref = "\n".join(nltk.sent_tokenize(ref))
        pred = "\n".join(nltk.sent_tokenize(pred))
        score = scorer.score(ref, pred)
        rouge1 += score['rouge1'].fmeasure
        rouge2 += score['rouge2'].fmeasure
        rougel += score['rougeL'].fmeasure
    print(f'Rouge scorer results: R-1 {rouge1 / len(all_results)}, R-2 {rouge2 / len(all_results)}, R-L {rougel / len(all_results)} ')

    os.system(f'./scripts/summarization/eval_rouge.sh {save_path} {raw_data_path}/{args.split}.tgt')

if __name__ == "__main__":
    main()
