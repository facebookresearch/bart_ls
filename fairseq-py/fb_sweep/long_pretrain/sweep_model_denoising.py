#!/usr/bin/env python

from fb_sweep import sweep
from fb_sweep.sweep import hyperparam

"""


python fb_sweep/long_pretrain/sweep_model_denoising.py -p md_vanilla_c4_r6 \
-g 8 -n 16 -t 1 --partition a100 --checkpoints-dir /data/home/xwhan/checkpoints/model_denoising --resume-failed \
    --baseline-model /data/home/xwhan/fairseq-py/checkpoints/md.base.16k.pool4.span3.r6/model.pt --snapshot-code \
        --local

"""

# shards on /fsx
# prefix = '/data/home/xwhan/data/pretrain_regimes/assembled_c4'

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
    num_data_loaders = 2

    arch = "loco_base"
    task = "model_based_denoising"
    criterion = "model_based_denoising"

    adam_eps = 1e-06
    weight_decay = 0.01
    lr = 1e-4

    sequence_len = 8192 * 2

    bsz = 8192 * 2 // sequence_len
    update_freq = 2

    grid += [
        hyperparam(
            "--custom-dict",
            # f'/data/home/xwhan/fairseq-py/checkpoints/base.md.8k.pool4.span5/dict.txt'
            '/data/home/xwhan/fairseq-py/checkpoints/md.base.16k.pool4.span3.r6/dict.txt'
        )
    ]

    # model settings
    grid += [
        hyperparam("--arch", arch, save_dir_key=lambda val: val),
        hyperparam("--task", task),
        hyperparam("--fast-stat-sync", save_dir_key=lambda _: "faststatsync"),
        hyperparam("--criterion", criterion),
        hyperparam("--required-seq-len-multiple", 1024),
        hyperparam("--use-xformers"),
        hyperparam("--attention-name", "block_noglobal", save_dir_key=lambda val: val),
        hyperparam("--pooling-layers", 4, save_dir_key=lambda val: f"pool{val}"),
        # hyperparam("--train-subset", "train" if not args.local else "valid"),
        hyperparam("--train-generator"),
        hyperparam("--generator-layers", 6),
    ]


    grid += [
        hyperparam("--max-source-positions", sequence_len, save_dir_key=lambda val: f"ms{val}"),
        hyperparam("--tokens-per-sample", sequence_len),
        hyperparam("--max-target-positions", 1024, save_dir_key=lambda val: f"mt{val}"),
        hyperparam("--update-freq", update_freq, save_dir_key=lambda val: f"uf{val}"),
        hyperparam(
            "--max-update", total_num_udpates, save_dir_key=lambda val: f"mu{val}"
        ),
        hyperparam("--required-batch-size-multiple", 1),
        hyperparam(
            "--sample-break-mode", ["complete"], save_dir_key=lambda val: f"brk_{val}"
        ),
    ]
    # regularization
    grid += [
        hyperparam("--dropout", 0.0, save_dir_key=lambda val: f"dr{val}"),
        hyperparam("--attention-dropout", 0.0, save_dir_key=lambda val: f"atdr{val}"),
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
        hyperparam("--mlm-loss-weight", 1, save_dir_key=lambda val: f"mlm{val}"),
        # hyperparam("--checkpoint-activations"),
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
        hyperparam("--log-interval", 1 if args.local else 20),
        hyperparam("--combine-val"),
        hyperparam("--save-interval-updates", 20000),
    ]
    grid += [
        hyperparam("--sample-ratio", 0.2, save_dir_key=lambda val: f"sample{val}"),
        hyperparam("--noise-density", 0.0625, save_dir_key=lambda val: f"noise{val}"),
        hyperparam("--dynamic-span-len", save_dir_key=lambda _: "dynaspan"),
        # hyperparam("--mean-noise-span-length", 5, save_dir_key=lambda val: f"spanlen{val}"),
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
