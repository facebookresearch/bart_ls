# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import argparse
from tqdm import tqdm
from fairseq import checkpoint_utils
from fairseq.tasks.summarization import load_langpair_dataset, load_query_based_dataset
from fairseq.models.bart.hub_interface import BARTHubInterface
from fairseq.data.encoders.gpt2_bpe import GPT2BPE, GPT2BPEConfig



from nltk import word_tokenize
# tokneize a sent
def tokenize(sent):
    tokens = ' '.join(word_tokenize(sent.lower()))
    return tokens


def main():
    """
    Usage::
         python scripts/summarization/qmsum_generate.py \
            --model-dir /checkpoints/xwhan/qmsum/qmsum_best_r3f.noiseuniform.r3f0.01.bart_large.mep150.sl16384.ls0.1.mq45.pad_q0.mixed.pool4.block_noglobal.mt2.uf2.dr0.1.atdr0.1.actdr0.0.wd0.01.s3.adam.beta9999.eps1e-08.clip0.1.lr5e-05.mu8000.warm200.fp16.cmetricrouge_avg.ngpu8/checkpoint_best.pt \
            --data-dir /fsx/xwhan/data/QMSum/data/raw-bin \
            --save-dir /fsx/xwhan/data/QMSum/data/raw/valid.hypo \
            --split valid \
            --bsz 4 

    SCROLLS submission:

         python scripts/summarization/qmsum_generate.py \
            --model-dir /checkpoints/xwhan/qmsum/qmsum_best_r3f.noiseuniform.r3f0.01.bart_large.mep150.sl16384.ls0.1.mq45.pad_q0.mixed.pool4.block_noglobal.mt2.uf2.dr0.1.atdr0.1.actdr0.0.wd0.01.s5.adam.beta9999.eps1e-08.clip0.1.lr5e-05.mu8000.warm100.fp16.cmetricrouge_avg.ngpu8/checkpoint_best.pt \
            --data-dir /fsx/xwhan/data/scrolls/qmsum/bin \
            --save-dir /fsx/xwhan/data/scrolls/qmsum/test.best \
            --split test \
            --bsz 4 \
            --skip-eval


    """
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
    parser.add_argument("--max-len", default=256, help="max_len_b", type=int)
    parser.add_argument(
        "--n", default=None, help="how many examples to summarize", type=int
    )
    parser.add_argument("--split", default='test', type=str)
    parser.add_argument("--skip-eval", action='store_true', default=False)

    args = parser.parse_args()

    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([args.model_dir])
    model = models[0]

    # task.cfg.required_seq_len_multiple = 1024
    # task.cfg.left_pad_source = False

    if not task.cfg.query_based:
        dataset_for_inference = load_langpair_dataset(args.data_dir, 
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
            )
    else:
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
    GEN_KWARGS = dict(beam=4, max_len_b=args.max_len, lenpen=4.0, no_repeat_ngram_size=2, min_len=40, patience_factor=1.0)

    print(GEN_KWARGS)

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
                **GEN_KWARGS
            )
            batch_outputs = [hub_interface.decode(hypos[0]["tokens"]) for hypos in batch_hypos]
            all_results.extend(batch_outputs)
    
    import datetime
    # suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    # save_path = args.save_dir + '.' + suffix
    save_path = args.save_dir

    with open(save_path, 'w') as out:
        for l in all_results:
            out.write(l.strip() + '\n')

    if args.skip_eval:
        return
    
    # sanity check
    from rouge_score import rouge_scorer
    import nltk


    raw_data_path = args.data_dir[:-len("-bin")]

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


    import os
    os.system(f'./scripts/summarization/eval_rouge.sh {save_path} {raw_data_path}/{args.split}.target')


if __name__ == "__main__":
    main()
