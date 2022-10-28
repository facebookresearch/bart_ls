#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""


python fb_sweep/long_finetune/sweep_summ.py -p ss_large_all \
-d /fsx/xwhan/data/summscreen/fd-bin \
-g 8 -n 1 -t -1 --partition a100 --checkpoints-dir /fsx/xwhan/checkpoints/summscreen --resume-failed --snapshot-code \
--baseline-model /fsx/xwhan/checkpoints/long_denoising/t5_all_corpus.bart_large.faststatsync.pool4.block_noglobal.ms16384.mt1024.uf1.mu500000.brk_complete.dr0.1.atdr0.1.actdr0.0.wd0.01.bsz4.adam.eps1e-06.clip0.1.s42.lr0.0001.warm500.memfp16.noise0.0625.dynaspan.ngpu128/model_100k.pt --no-wandb --local

"""
from fb_sweep import sweep
from fb_sweep.sweep import hyperparam


def get_grid(args):
    grid = []

    total_num_udpates = 10000
    warmup_updates = 200
    num_data_loaders = 4
    # arch = "bart_base"

    arch = 'bart_large'
    task = "summarization"
    criterion = "label_smoothed_cross_entropy"

    lrs = [3e-4, 1e-4]
    source, target = 'src', 'tgt'
    dropout = 0

    if 'base' in arch:
        bsz = 4
        update_freq = 2
        grid += [
            hyperparam("--checkpoint-activations"),
        ]
    else:
        bsz, update_freq = 2, 4
        grid += [
            hyperparam("--checkpoint-activations"),
        ]

    if 'arxiv' in args.data:
        lrs = [3e-4, 1e-4, 4e-4]
        max_epoch = 8 # arxiv
        generate_args = '{"beam": 4, "max_len_b": 300, "lenpen": 2.0, "no_repeat_ngram_size": 3, "min_len": 20}'
    elif 'tv' in args.data:
        total_num_udpates = 15000
        dropout = [0]
        max_epoch = 60
        source, target = 'source', 'target'
        generate_args = '{"beam": 4, "max_len_b": 700, "lenpen": 2.0, "no_repeat_ngram_size": 3, "min_len": 20}'
        bsz, update_freq = 2, 2
    elif 'summscreen' in args.data:
        lrs = [5e-5, 3e-5]
        warmup_updates = [200, 500, 1000]
        max_epoch = 130
        dropout = [0, 0.2]
        generate_args = '{"beam": 4, "max_len_b": 300, "lenpen": 2.0, "no_repeat_ngram_size": 3, "min_len": 20}'
    elif 'gov_report' in args.data:
        lrs = [3e-4, 5e-5, 4e-4]
        total_num_udpates = 15000
        max_epoch = 70
        generate_args = '{"beam": 4, "max_len_b": 450, "lenpen": 2.0, "no_repeat_ngram_size": 3, "min_len": 60}'
    elif 'booksum' in args.data:
        max_epoch = 60
        update_freq = 4
        generate_args = '{"beam": 4, "max_len_b": 450, "lenpen": 4.0, "no_repeat_ngram_size": 3, "min_len": 20}'
    elif 'pubmed' in args.data:
        lrs = [1e-4]
        max_epoch = 12 # TODO needs to be increased for better performance from 10
        generate_args = '{"beam": 4, "max_len_b": 400, "lenpen": 2.0, "no_repeat_ngram_size": 3, "min_len": 20}'

    else:
        assert False, "Max epoch not set for this dataset"

    adam_eps = 1e-08
    max_source_positions = 1024*16

    # model to use
    grid += [
        # hyperparam(
        #     "--restore-file",
        #     f"{pretrain_path}seq2seq_100k.pt",
        #     # "/data/home/xwhan/fairseq-py/checkpoints/bart.large.block16k/model.pt"
        # ),
        hyperparam(
            "--custom-dict",
            # f'/data/home/xwhan/fairseq-py/checkpoints/bart.base.block8k.pool.t5/dict.txt' # t5 pretrain
            "/data/home/xwhan/fairseq-py/checkpoints/bart.large.block16k.pool.t5.span3/dict.txt",
            # "/checkpoints/xwhan/model_denoising/md_joint.loco_base.faststatsync.block_sw.pool4.ms8192.mt1024.uf4.mu100000.brk_complete_doc.dr0.1.atdr0.1.actdr0.0.wd0.01.bsz1.adam.beta9999.eps1e-06.clip0.1.s42.lr0.0001.warm500.memfp16.sample0.2.noise0.1.spanlen5.ngpu64/dict.txt"
            # f'{pretrain_path}/dict.txt'
            # "/data/home/xwhan/fairseq-py/checkpoints/bart.base.block16k.pool/dict.txt"
        )
    ]


    # grid += [
    #     hyperparam("--ddp-backend", "no_c10d"),
    # ]


    # model settings
    grid += [
        hyperparam("--arch", arch, save_dir_key=lambda val: val),
        # hyperparam("--train-subset", "train" if not args.local else "valid"),
        hyperparam("--task", task),
        hyperparam("--required-seq-len-multiple", 1024),
        hyperparam("--criterion", criterion),
        hyperparam("--max-epoch", max_epoch, save_dir_key=lambda val: f"mep{val}"),
        hyperparam("--max-source-positions", max_source_positions, save_dir_key=lambda val: f"sl{val}"),
        hyperparam("--max-target-positions", 1024),
        hyperparam("--source-lang", source),
        hyperparam("--target-lang", target),
        hyperparam("--truncate-source"),
        hyperparam("--truncate-target"),
        hyperparam("--label-smoothing", 0.1, save_dir_key=lambda val: f"ls{val}"),
        hyperparam("--pooling-layers", 4, save_dir_key=lambda val: f"pool{val}"),
        hyperparam("--use-xformers"),
        hyperparam("--attention-name", ['block_noglobal'], save_dir_key=lambda val: val),
    ]

    grid += [
        hyperparam("--batch-size", bsz, save_dir_key=lambda val: f"mt{val}"),
        hyperparam("--batch-size-valid", 2 if 'gov' in args.data else 4),
        hyperparam("--update-freq", update_freq, save_dir_key=lambda val: f"uf{val}"),
        hyperparam("--required-batch-size-multiple", 1),
    ]
    # regularization
    grid += [
        hyperparam("--dropout", dropout, save_dir_key=lambda val: f"dr{val}"),
        hyperparam("--attention-dropout", [0.1, 0], save_dir_key=lambda val: f"atdr{val}"),
        hyperparam("--relu-dropout", 0.0, save_dir_key=lambda val: f"actdr{val}"),
        hyperparam("--weight-decay", 0.01, save_dir_key=lambda val: f"wd{val}"),
    ]

    # optimization settings
    grid += [
        hyperparam("--seed", [3, 42], save_dir_key=lambda val: f"s{val}"),
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--adam-betas", "(0.9, 0.999)", save_dir_key=lambda val: "beta9999"),
        hyperparam("--adam-eps", adam_eps, save_dir_key=lambda val: f"eps{val}"),
        hyperparam("--clip-norm", [0.1, 0], save_dir_key=lambda val: f"clip{val}"),
    ]

    # lr schedule4
    grid += [
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam("--lr", lrs, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--total-num-update", total_num_udpates, save_dir_key=lambda val: f"mu{val}"),
        hyperparam(
            "--warmup-updates", warmup_updates, save_dir_key=lambda val: f"warm{val}"
        ),
    ]
    grid += [
        hyperparam("--memory-efficient-fp16", save_dir_key=lambda val: "memfp16"),
    ]

    # data loading settings
    grid += [
        hyperparam("--num-workers", num_data_loaders),
    ]

    valid_interval_updates = 500 if ('arxiv' in args.data or 'pubmed' in args.data or 'govreport' in args.data) else 200

    # validation and checkpoint settings
    grid += [
        hyperparam("--validate-interval", int(max_epoch // 5)),
        hyperparam("--no-epoch-checkpoints"),
        hyperparam("--validate-interval-updates", 10 if args.local else valid_interval_updates),
        hyperparam("--best-checkpoint-metric", "rouge_avg", save_dir_key=lambda val: f"cmetric{val}")
    ]

    # logging settings
    grid += [
        hyperparam("--skip-invalid-size-inputs-valid-test"),
        hyperparam("--maximize-best-checkpoint-metric"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 10),
        hyperparam("--eval-rouge"),
        hyperparam("--eval-rouge-args", generate_args),
    ]

    if args.local:
        grid += [
            hyperparam("--log-format", "json"),
            hyperparam("--log-interval", 1),
        ]
    return grid


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
