# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# from datasets import load_dataset
import torch
import argparse
from tqdm import tqdm
from fairseq import checkpoint_utils
from fairseq.tasks.summarization import load_query_based_dataset
from fairseq.models.bart.hub_interface import BARTHubInterface
from fairseq.data.encoders.gpt2_bpe import GPT2BPE, GPT2BPEConfig
from collections import defaultdict
from fairseq.tasks.qa import f1_score, exact_match_score
import numpy as np
import json

"""
Aggregate predictions for same questions, and use the maximum f1/EM scores
"""

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    assert isinstance(prediction, str)
    assert isinstance(ground_truths, list)
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

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
    parser.add_argument("--bsz", default=1, help="batch size", type=int)
    parser.add_argument("--data-name", default='qasper', type=str)
    parser.add_argument(
        "--n", default=None, help="how many examples to summarize", type=int
    )
    parser.add_argument("--split", default='test', type=str)
    parser.add_argument("--combine-instances", action='store_true', default=False)
    parser.add_argument("--json-gold", action='store_true', default=False, help='load json format groundtruth')
    parser.add_argument("--save-dir", default=None, type=str)
    parser.add_argument("--skip-eval", action='store_true', default=False)
    args = parser.parse_args()

    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([args.model_dir])
    model = models[0]

    generate_args_narrative = dict(beam=4,  max_len_b=20, lenpen=1.0, no_repeat_ngram_size=3)
    generate_args_qasper = dict(beam=4,  max_len_b=80, lenpen=1.0, no_repeat_ngram_size=3)
    generate_args_quality = dict(beam=4,  max_len_b=50, lenpen=3.0, no_repeat_ngram_size=3)
    if 'narrative' in args.data_name:
        generate_args = generate_args_narrative
    elif 'quality' in args.data_name:
        generate_args = generate_args_quality
    elif 'contract' in args.data_name:
        generate_args = dict(beam=4,  max_len_b=6, lenpen=3.0, no_repeat_ngram_size=3)
    else:
        generate_args = generate_args_qasper

    print(f'generating arguments {generate_args}')


    task.cfg.required_seq_len_multiple = 1024
    task.cfg.left_pad_source = False
    dataset_for_inference = load_query_based_dataset(args.data_dir, 
        split=args.split,
        src='source',
        src_dict=task.src_dict,
        tgt='target',
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
        max_query_positions=task.cfg.max_query_positions,
        input_pattern=task.cfg.input_pattern,
        blocksize=task.cfg.block_size,
        qry=task.cfg.query_lang,
        pad_q_len=task.cfg.pad_query
        )
    
    hub_interface = BARTHubInterface(cfg, task, model)
    hub_interface.bpe = GPT2BPE(GPT2BPEConfig)
    
    # HACK
    hub_interface.cfg.dataset.batch_size = args.bsz
    all_sents = [item['source'] for item in dataset_for_inference]

    all_results = []
    with torch.no_grad():
        hub_interface = hub_interface.eval()
        assert torch.cuda.is_available()
        hub_interface = hub_interface.cuda().half()
        for idx in tqdm(range(0, len(all_sents), args.bsz)):
            batch_sents = all_sents[idx:idx+args.bsz]
            batch_hypos = hub_interface.generate(
                batch_sents,
                skip_invalid_size_inputs=False,
                **generate_args
            )
            batch_outputs = [hub_interface.decode(hypos[0]["tokens"]) for hypos in batch_hypos]
            all_results.extend(batch_outputs)

    if args.save_dir:
        with open(args.save_dir, 'w') as out:
            for l in all_results:
                out.write(l.strip() + '\n')

    if args.skip_eval:
        return
        
    # load groundtruth
    raw_data_path = args.data_dir[:-len("-bin")]
    split = 'val' if args.split == 'valid' else 'test'
    questions = [l.strip() for l in open(f'{raw_data_path}/{split}.query').readlines()]
    answers = [l.strip() for l in open(f'{raw_data_path}/{split}.target').readlines()]
    inputs = [l.strip() for l in open(f'{raw_data_path}/{split}.source').readlines()]


    predictions = []
    golds = []

    if args.combine_instances:
        last_question = None
        last_input = None
        curr_gold = []
        for q, pred, gold, doc in zip(questions, all_results, answers, inputs):
            if q != last_question or (last_input is None or doc[:1000] != last_input[:1000]):
                predictions.append(pred)
                if len(curr_gold) > 0:
                    golds.append(curr_gold)
                    curr_gold = []
            curr_gold.append(gold)
            last_question = q
            last_input = doc

        if curr_gold:
            golds.append(curr_gold)
    else:
        for pred, gold in zip(all_results, answers):
            predictions.append(pred)
            if args.json_gold:
                golds.append(json.loads(gold))
            else:
                golds.append([gold])

    ems, f1s = [], []
    for pred, grounds in zip(predictions, golds):
        ems.append(metric_max_over_ground_truths(exact_match_score, pred, grounds))
        f1s.append(metric_max_over_ground_truths(f1_score, pred, grounds))

    print(f'Mean EM: {np.mean(ems)}, f1: {np.mean(f1s)} of {len(ems)} examples on {split}')

if __name__ == "__main__":
    main()
