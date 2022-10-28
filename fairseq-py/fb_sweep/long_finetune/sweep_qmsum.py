# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python

from fb_sweep import sweep
from fb_sweep.sweep import hyperparam

"""




python fb_sweep/long_finetune/sweep_qmsum.py -p qmsum_best_r3f \
    -d /fsx/xwhan/data/QMSum/data/raw-bin \
-g 8 -n 1 -t -1 --partition hipri --checkpoints-dir /checkpoints/xwhan/qmsum --resume-failed --snapshot-code \
--baseline-model /fsx/xwhan/checkpoints/long_denoising/t5_all_corpus.bart_large.faststatsync.pool4.block_noglobal.ms16384.mt1024.uf1.mu500000.brk_complete.dr0.1.atdr0.1.actdr0.0.wd0.01.bsz4.adam.eps1e-06.clip0.1.s42.lr0.0001.warm500.memfp16.noise0.0625.dynaspan.ngpu128/model_100k.pt  --local --time 1440



### test model denoising

python fb_sweep/long_finetune/sweep_qmsum.py -p qmsum_md \
    -d /fsx/xwhan/data/QMSum/data/raw-bin \
-g 8 -n 1 -t -1 --partition lowpri --checkpoints-dir /checkpoints/xwhan/qmsum --resume-failed --snapshot-code \
--baseline-model /data/home/xwhan/checkpoints/model_denoising/md_assembled_c4.loco_base.faststatsync.block_noglobal.pool4.ms16384.mt1024.uf2.mu100000.brk_complete.dr0.0.atdr0.0.actdr0.0.wd0.01.bsz1.adam.beta9999.eps1e-06.clip0.1.s42.lr0.0001.warm500.memfp16.sample0.2.noise0.0625.dynaspan.ngpu128/model_100k.pt --time 1440
"""



def get_grid(args):
    grid = []

    # total_num_udpates = 5000

    total_num_udpates = 8000 # larger for r3f
    warmup_updates = [100, 200]
    num_data_loaders = 4
    arch = "bart_large"
    task = "summarization"
    criterion = "label_smoothed_cross_entropy"

    adam_eps = 1e-08

    max_source_positions = 1024*16
    max_epoch = 150

    # which model to use
    grid += [
        # hyperparam("--train-subset", "train" if not args.local else "valid"),
        #     # "/checkpoints/xwhan/model_denoising/md_joint_g512.loco_large.ms8192.ts8192.mt1024.uf2.mu100000.brk_complete.dr0.1.atdr0.1.actdr0.0.wd0.01.bsz4.adam.beta9999.eps1e-06.clip0.1.s42.lr3e-05.warm500.memfp16.sample0.2.noise0.1.ngpu32/seq2seq_100k.pt",
        # ),
        # hyperparam(
        #     "--custom-dict",
        #     "/checkpoints/xwhan/model_denoising/md_joint_pool.loco_large.pool4.ms8192.ts8192.mt1024.uf1.mu100000.brk_complete.dr0.1.atdr0.1.actdr0.0.wd0.01.bsz4.adam.beta9999.eps1e-06.clip0.1.s42.lr3e-05.warm500.memfp16.sample0.2.noise0.1.ngpu64/dict.txt"
        #     # f'/checkpoints/xwhan/model_denoising/md_joint_g512.loco_large.ms8192.ts8192.mt1024.uf2.mu100000.brk_complete.dr0.1.atdr0.1.actdr0.0.wd0.01.bsz4.adam.beta9999.eps1e-06.clip0.1.s42.lr3e-05.warm500.memfp16.sample0.2.noise0.1.ngpu32/dict.txt'
        # )
        hyperparam(
            "--custom-dict",
            # f'/data/home/xwhan/fairseq-py/checkpoints/bart.base.block16k/dict.txt',
            # '/data/home/xwhan/fairseq-py/checkpoints/md.base.16k.pool4.span3.a6/dict.txt'
            '/data/home/xwhan/fairseq-py/checkpoints/bart.large.block16k.pool.t5.span3/dict.txt'
        )
    ]


    if 'base' in arch:
        bsz = 4
        update_freq = 2
        grid += [
            hyperparam("--checkpoint-activations"),
        ]
    else:
        bsz, update_freq = 4, 1
        grid += [
            hyperparam("--checkpoint-activations"),
        ]

    # better finetuning
    criterion = "label_smoothed_cross_entropy_r3f"
    bsz = bsz//2
    update_freq = update_freq*2
    grid += [
        hyperparam("--noise-type", ["uniform"], save_dir_key=lambda val: f"noise{val}"),
        hyperparam("--r3f-lambda", [0.01], save_dir_key=lambda val: f"r3f{val}"),
        hyperparam("--user-dir", "examples/rxf/rxf_src"),
        hyperparam("--ddp-backend", "no_c10d"),
        hyperparam("--reset-optimizer"),
    ]

    # model settings
    grid += [
        hyperparam("--arch", arch, save_dir_key=lambda val: val),
        hyperparam("--required-seq-len-multiple", 1024),
        hyperparam("--task", task),
        hyperparam("--criterion", criterion),
        hyperparam("--max-epoch", max_epoch, save_dir_key=lambda val: f"mep{val}"),
        hyperparam("--max-source-positions", max_source_positions, save_dir_key=lambda val: f"sl{val}"),
        hyperparam("--max-target-positions", 1024),
        hyperparam("--source-lang", "source"),
        hyperparam("--target-lang", "target"),
        hyperparam("--truncate-source"),
        hyperparam("--label-smoothing", 0.1, save_dir_key=lambda val: f"ls{val}"),
        hyperparam("--query-based"),
        hyperparam("--max-query-positions", 45, save_dir_key=lambda val: f"mq{val}"),
        hyperparam("--pad-query", 0, save_dir_key=lambda val: f"pad_q{val}"),
        hyperparam("--input-pattern", ['mixed'], save_dir_key=lambda val: f"{val}"),
        hyperparam("--use-xformers"),
        hyperparam("--pooling-layers",4, save_dir_key=lambda val: f"pool{val}"),
        hyperparam("--attention-name", ['block_noglobal'], save_dir_key=lambda val: val),
    ]

    grid += [
        hyperparam("--batch-size", bsz, save_dir_key=lambda val: f"mt{val}"),
        hyperparam("--batch-size-valid", 4),
        hyperparam("--update-freq", update_freq, save_dir_key=lambda val: f"uf{val}"),
        hyperparam("--required-batch-size-multiple", 1),
    ]
    # regularization
    grid += [
        hyperparam("--dropout", 0.1, save_dir_key=lambda val: f"dr{val}"),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"atdr{val}"),
        hyperparam("--relu-dropout", 0.0, save_dir_key=lambda val: f"actdr{val}"),
        hyperparam("--weight-decay", 0.01, save_dir_key=lambda val: f"wd{val}"),
    ]

    # optimization settings
    grid += [
        hyperparam("--seed", [3, 5], save_dir_key=lambda val: f"s{val}"),
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--adam-betas", "(0.9, 0.999)", save_dir_key=lambda val: "beta9999"),
        hyperparam("--adam-eps", adam_eps, save_dir_key=lambda val: f"eps{val}"),
        hyperparam("--clip-norm", 0.1, save_dir_key=lambda val: f"clip{val}"),
    ]

    # lr scheduler
    grid += [
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam("--lr", [5e-5, 1e-4], save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--total-num-update", total_num_udpates, save_dir_key=lambda val: f"mu{val}"),
        hyperparam(
            "--warmup-updates", warmup_updates, save_dir_key=lambda val: f"warm{val}"
        ),
    ]
    grid += [
        hyperparam("--memory-efficient-fp16", save_dir_key=lambda val: "fp16"),
    ]

    # data loading settings
    grid += [
        hyperparam("--num-workers", num_data_loaders),
    ]

    # validation and checkpoint settings
    grid += [
        hyperparam("--validate-interval", int(max_epoch // 5)),
        hyperparam("--no-epoch-checkpoints"),
        hyperparam("--validate-interval-updates", 200 if not args.local else 10),
        hyperparam("--best-checkpoint-metric", "rouge_avg", save_dir_key=lambda val: f"cmetric{val}")
    ]


    # logging settings
    grid += [
        hyperparam("--skip-invalid-size-inputs-valid-test"),
        hyperparam("--maximize-best-checkpoint-metric"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 10),
        hyperparam("--eval-rouge"),
        hyperparam("--eval-rouge-args", '{"beam": 4, "max_len_b": 256, "lenpen": 2.0, "no_repeat_ngram_size": 3, "min_len": 20}'),
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
