# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import argparse
from tqdm import tqdm
from fairseq import checkpoint_utils
from fairseq.tasks.summarization import load_langpair_dataset
from fairseq.models.bart.hub_interface import BARTHubInterface
from fairseq.data.encoders.gpt2_bpe import GPT2BPE, GPT2BPEConfig
import nltk


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
    parser.add_argument("--skip-eval", action='store_true', default=False)

    args = parser.parse_args()

    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([args.model_dir])
    model = models[0]

    print(f"Num of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


    task.cfg.required_seq_len_multiple = 1024
    task.cfg.left_pad_source = False
    dataset_for_inference = load_langpair_dataset(args.data_dir, 
        split=args.split,
        src='src',
        src_dict=task.src_dict,
        tgt='tgt',
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
        pad_to_multiple=task.cfg.required_seq_len_multiple,
        )
    
    hub_interface = BARTHubInterface(cfg, task, model)
    hub_interface.bpe = GPT2BPE(GPT2BPEConfig)
    
    hub_interface.cfg.dataset.batch_size = args.bsz
    all_sents = [item['source'] for item in dataset_for_inference]
    
    if 'gov_report' in args.data_dir:
        generate_args = dict(beam=4,  max_len_b=1024, lenpen=3.0, no_repeat_ngram_size=3, min_len=60)
    elif 'summscreen' in args.data_dir:
        generate_args = dict(beam=4, max_len_b=350, lenpen=2.0, no_repeat_ngram_size=3, min_len=50)
    elif 'booksum' in args.data_dir:
        generate_args = dict(beam=4, max_len_b=320, lenpen=2.0, no_repeat_ngram_size=4, min_len=20)
    else:
        generate_args = dict(beam=4, max_len_b=256, lenpen=2.0, no_repeat_ngram_size=4, min_len=50)

    print(f'Generating parameters {generate_args}')

    all_results = []
    with torch.no_grad():
        hub_interface = hub_interface.eval()
        assert torch.cuda.is_available()
        hub_interface = hub_interface.cuda()
        for idx in tqdm(range(0, len(all_sents), args.bsz)):
            batch_sents = all_sents[idx:idx+args.bsz]
            batch_hypos = hub_interface.generate(
                batch_sents,
                skip_invalid_size_inputs=False,
                **generate_args
            )
            batch_outputs = [hub_interface.decode(hypos[0]["tokens"]) for hypos in batch_hypos]
            all_results.extend(batch_outputs)

    save_path = args.save_dir

    with open(save_path, 'w') as out:
        for l in all_results:
            out.write(l.strip() + '\n')

    if args.skip_eval:
        return
    
    raw_data_path = args.data_dir[:-len("-bin")]

    from rouge_score import rouge_scorer
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
    
    print(f'Rouge (by rouge_score) results: R-1 {rouge1 / len(all_results)}, R-2 {rouge2 / len(all_results)}, R-L {rougel / len(all_results)} ')

    save_path = args.save_dir

    with open(save_path, 'w') as out:
        for l in all_results:
            out.write(l.strip() + '\n')

if __name__ == "__main__":
    main()
