#!/usr/bin/env python

from fb_sweep import sweep
from fb_sweep.sweep import hyperparam


"""

python fb_sweep/sweep_compare_position_encoding.py -p bart_alibi \
-g 1 -n 1 -t -1 --partition learnfair --checkpoints-dir /data/home/xwhan/checkpoints/pos --resume-failed --tensorboard-logdir /data/home/xwhan/checkpoints/pos/tb --snapshot-code --local

python fb_sweep/sweep_compare_position_encoding.py -p bart_vanilla \
-d /fsx/xwhan/data/pretrain_corpus/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin/121219/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin \
-g 8 -n 8 -t -1 --partition a100 --checkpoints-dir /data/home/xwhan/checkpoints/pos --resume-failed

"""


SHARDS = [
    '/data/home/xwhan/data/pretrain_regimes/roberta_data/shard0/',
    '/data/home/xwhan/data/pretrain_regimes/roberta_data/shard1/',
    '/data/home/xwhan/data/pretrain_regimes/roberta_data/shard2/',
    '/data/home/xwhan/data/pretrain_regimes/roberta_data/shard3/',
    '/data/home/xwhan/data/pretrain_regimes/roberta_data/shard4/'
]

def get_grid(args):
    grid = []

    total_num_udpates = 600000
    warmup_updates = int(total_num_udpates * 0.06)
    num_data_loaders = 2
    arch = "bart_large"
    task = "denoising"
    criterion = "cross_entropy"

    adam_eps = 1e-06
    weight_decay = 0.01

    update_freq = 4

    # model settings
    grid += [
        hyperparam("--arch", arch, save_dir_key=lambda val: val),
        hyperparam("--task", task),
        hyperparam("--criterion", criterion),
        hyperparam("--train-subset", "train" if not args.local else "valid"),
        # hyperparam("--use-xformers"),
        # hyperparam("--attention-name", 'block_noglobal'),
        # hyperparam("--xformer-config", '{"block_size": 1024}'),
        hyperparam("--use-xformers"),
        hyperparam("--attention-name", "bs_local"),
        hyperparam("--xformer-config", '{"block_size": 1024, "max_seq_len": 1024}'),
        hyperparam("--alibi", save_dir_key=lambda val: "alibi")
    ]

    # grid += [
    #     hyperparam("--encoder-normalize-before", save_dir_key=lambda val: "enb"),
    #     hyperparam("--decoder-normalize-before", save_dir_key=lambda val: "dnb"),
    # ]

    # grid += [
    #     hyperparam('--ddp-backend', 'fully_sharded'),
    #     hyperparam('--no-reshard-after-forward'),
    # ]

    grid += [
        hyperparam("--max-source-positions", 1024),
        hyperparam("--max-target-positions", 1024),
        hyperparam("--tokens-per-sample", 512),
        hyperparam("--batch-size", 16, save_dir_key=lambda val: f"ms{val}"),
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
        hyperparam("--dropout", 0.1, save_dir_key=lambda val: f"dr{val}"),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"atdr{val}"),
        hyperparam("--relu-dropout", 0.0, save_dir_key=lambda val: f"actdr{val}"),
        hyperparam("--weight-decay", weight_decay, save_dir_key=lambda val: f"wd{val}"),
    ]

    # optimization settings
    grid += [
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--adam-betas", "(0.9, 0.98)", save_dir_key=lambda val: "beta9999"),
        hyperparam("--adam-eps", adam_eps, save_dir_key=lambda val: f"eps{val}"),
        hyperparam("--clip-norm", 0.1, save_dir_key=lambda val: f"clip{val}"),
        hyperparam("--combine-val")
    ]

    # lr scheduler
    grid += [
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam("--lr", 7e-04, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--total-num-update", total_num_udpates),
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
        hyperparam("--no-epoch-checkpoints"),
    ]

    # logging settings
    grid += [
        hyperparam("--skip-invalid-size-inputs-valid-test"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 20),
    ]
    grid += [
        hyperparam("--poisson-lambda", 3.5, save_dir_key=lambda val: f"poi_lam{val}"),
        hyperparam("--mask", 0.3, save_dir_key=lambda val: f"mask{val}"),
        hyperparam(
            "--mask-length", "span-poisson", save_dir_key=lambda val: f"mask_len{val}"
        ),
        hyperparam("--replace-length", 1, save_dir_key=lambda val: f"rpl_len{val}"),
        hyperparam("--rotate", 0, save_dir_key=lambda val: f"rotate{val}"),
        hyperparam("--mask-random", 0.1, save_dir_key=lambda val: f"mask_rand{val}"),
        hyperparam("--insert", 0, save_dir_key=lambda val: f"ins{val}"),
        hyperparam(
            "--permute-sentences", 1.0, save_dir_key=lambda val: f"perm_sen{val}"
        ),
    ]

    if args.local:
        grid += [
            hyperparam("--log-format", "json"),
            hyperparam("--log-interval", 1),
        ]
    return grid


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    args.data = ':'.join(SHARDS)


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
