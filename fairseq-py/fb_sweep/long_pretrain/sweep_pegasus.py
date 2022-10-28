# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python

from fb_sweep import sweep
from fb_sweep.sweep import hyperparam


"""
python fb_sweep/long_pretrain/sweep_pegasus.py -p pegasus_vanilla_c4 \
-g 8 -n 8 -t -1 --partition a100 --checkpoints-dir /fsx/xwhan/checkpoints/pegasus --resume-failed \
--baseline-model /data/home/xwhan/fairseq-py/checkpoints/bart.base.block16k.pool/model.pt \
--local

"""

# # shards on /data
# prefix = '/data/home/xwhan/data'

prefix = '/fsx/xwhan/data/pretrain_corpus/pretrain_regimes/c4_10shards'



SHARDS = [
    f'{prefix}/shard0/',
    f'{prefix}/shard1/',
    f'{prefix}/shard2/',
    f'{prefix}/shard3/',
    f'{prefix}/shard4/',
    f'{prefix}/shard5/',
    f'{prefix}/shard6/',
    f'{prefix}/shard7/',
    f'{prefix}/shard8/',
    f'{prefix}/shard9/',
]


def get_grid(args):
    grid = []

    total_num_udpates = 500000
    warmup_updates = 500
    num_data_loaders = 8

    arch = "bart_base"
    task = "pegasus"
    criterion = "cross_entropy"

    adam_eps = 1e-06
    weight_decay = 0.01
    lr = 1e-4

    sequence_len = 8192 * 2

    bsz = 8192 * 2 // sequence_len
    update_freq = 2

    # grid += [
    #     hyperparam(
    #         "--restore-file",
    #         "/data/home/xwhan/fairseq-py/checkpoints/bart_block8k_pool/model.pt",
    #         # "/data/home/xwhan/fairseq-py/checkpoints/bart.large.block8k/model.pt",
    #         # "/checkpoints/xwhan/pegasus/pegasus.bart_large.mt1024.msp8192.mtp1024.uf8.mu100000.brk_complete.dr0.1.atdr0.1.actdr0.0.wd0.01.bsz1.adam.beta9999.eps1e-06.clip0.1.s42.lr3e-05.warm500.memfp16.ngpu32/checkpoint_best.pt"
    #     ),
    # ]

    grid += [
        hyperparam(
            "--custom-dict",
            f'/data/home/xwhan/fairseq-py/checkpoints/bart.base.block16k.pool/dict.txt'
        )
    ]

    # model settings
    grid += [
        hyperparam("--arch", arch, save_dir_key=lambda val: val),
        hyperparam("--fast-stat-sync", save_dir_key=lambda _: "faststatsync"),
        hyperparam("--task", task),
        hyperparam("--required-seq-len-multiple", 1024),
        hyperparam("--criterion", criterion),
        hyperparam("--pooling-layers", 4, save_dir_key=lambda val: f"pool{val}"),
        hyperparam("--use-xformers"),
        hyperparam("--attention-name", "block_noglobal", save_dir_key=lambda val: val),
        # hyperparam("--xformer-config", '{"window_size": 512}'),
        # hyperparam("--attention-name", "bs_local", save_dir_key=lambda val: val),
        # hyperparam("--xformer-config", '{"block_size": 1024, "max_seq_len": 8192}'),
        hyperparam("--train-subset", "train" if not args.local else "valid"),
    ]

    # task settings
    grid += [
        hyperparam("--max-target-len", 1024, save_dir_key=lambda val: f"mt{val}"),
        hyperparam("--truncate-target"),
    ]

    grid += [
        hyperparam("--max-source-positions", sequence_len, save_dir_key=lambda val: f"msp{val}"),
        hyperparam("--tokens-per-sample", sequence_len, save_dir_key=lambda val: f"tps{val}"),
        hyperparam("--max-target-positions", 1024, save_dir_key=lambda val: f"mtp{val}"),
        
        hyperparam("--update-freq", update_freq, save_dir_key=lambda val: f"uf{val}"),
        hyperparam(
            "--max-update", total_num_udpates, save_dir_key=lambda val: f"mu{val}"
        ),
        hyperparam(
            "--sample-break-mode", ["complete"], save_dir_key=lambda val: f"brk_{val}"
        ),
        hyperparam("--mask-ratio", 0.0625, save_dir_key=lambda val: f"ms{val}")
    ]
    # regularization
    grid += [
        hyperparam("--dropout", 0.1, save_dir_key=lambda val: f"dr{val}"),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"atdr{val}"),
        hyperparam("--relu-dropout", 0.0, save_dir_key=lambda val: f"actdr{val}"),
        hyperparam("--weight-decay", weight_decay, save_dir_key=lambda val: f"wd{val}"),
    ]

    # optimization settings
    grid += [
        hyperparam("--batch-size", bsz, save_dir_key=lambda val: f"bsz{val}"),
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--adam-betas", "(0.9, 0.98)", save_dir_key=lambda val: "beta9999"),
        hyperparam("--adam-eps", adam_eps, save_dir_key=lambda val: f"eps{val}"),
        hyperparam("--clip-norm", 0.1, save_dir_key=lambda val: f"clip{val}"),
        hyperparam("--fp16-scale-tolerance", 0.1, save_dir_key=lambda val: f"fst{val}")
    ]

    # lr scheduler
    grid += [
        hyperparam("--seed", 42, save_dir_key=lambda val: f"s{val}"),
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam("--lr", lr, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--total-num-update", total_num_udpates),
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

    # validation and checkpoint settings
    grid += [
        hyperparam("--no-epoch-checkpoints"),
    ]

    # logging settings
    grid += [
        hyperparam("--skip-invalid-size-inputs-valid-test"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 10),
        hyperparam("--combine-val"),
        hyperparam("--save-interval-updates", 10000),
    ]


    if args.local:
        grid += [
            hyperparam("--log-format", "json"),
            hyperparam("--log-interval", 10),
        ]
    return grid


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    args.data = ':'.join(SHARDS)


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
