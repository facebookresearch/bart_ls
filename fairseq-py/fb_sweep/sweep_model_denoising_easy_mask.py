# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python

from fb_sweep import sweep
from fb_sweep.sweep import hyperparam



"""



python fb_sweep/sweep_model_denoising_easy_mask.py -p md_joint_g512 \
-d /fsx/xwhan/data/pretrain_corpus/long \
-g 8 -n 8 -t -1 --partition a100 --checkpoints-dir /checkpoints/xwhan/model_denoising --resume-failed --no-wandb


"""


def get_grid(args):
    grid = []

    total_num_udpates = 100000
    warmup_updates = 500
    num_data_loaders = 4

    arch = "loco_variant_large"
    task = "model_based_denoising"
    criterion = "model_based_denoising"

    adam_eps = 1e-06
    weight_decay = 0.01
    lr = 3e-5

    update_freq = 1
    grid += [
        hyperparam(
            "--restore-file",
            "/data/home/xwhan/fairseq-py/checkpoints/local_large_v0/model.pt",
            # "/checkpoints/xwhan/model_denoising/md_fixf_512.loco_large.ms8192.ts8192.mt1024.uf2.mu100000.brk_complete.dr0.1.atdr0.1.actdr0.0.wd0.01.bsz1.adam.beta9999.eps1e-06.clip0.1.s42.lr3e-05.warm500.memfp16.sample0.2.noise0.1.ngpu64/checkpoint_last.pt"
            # "/data/home/xwhan/fairseq-py/checkpoints/local_large_v0/model.pt",
            # "/data/home/xwhan/fairseq-py/checkpoints/loco_scratch_roberta/model.pt",
            # "/checkpoints/xwhan/model_denoising/model_denoisng_joint.loco_large.ms8192.ts8192.mt1024.uf1.mu100000.brk_complete.dr0.1.atdr0.1.actdr0.0.wd0.01.bsz4.adam.beta9999.eps1e-06.clip0.1.s42.lr3e-05.warm500.memfp16.sample0.2.noise0.1.ngpu32/checkpoint_last.pt"
            # "/data/home/xwhan/fairseq-py/checkpoints/local_large_v0/model.pt",
        ),
    ]

    # model settings
    grid += [
        hyperparam("--arch", arch, save_dir_key=lambda val: val),
        hyperparam("--task", task),
        hyperparam("--criterion", criterion),
        hyperparam("--use-xformers"),
        hyperparam("--attention-name", "bs_local"),
        hyperparam("--xformer-config", '{"block_size": 1024, "max_seq_len": 8192}'),
        hyperparam("--generator-xformer-config", '{"block_size": 512, "max_seq_len": 8192}'),
        hyperparam("--train-subset", "train" if not args.local else "valid"),
        hyperparam("--train-generator"),
        hyperparam("--easy-span-ops", ['sample'], save_dir_key=lambda val: f"sm_{val}"),
    ]

    grid += [
        hyperparam("--max-source-positions", 8192, save_dir_key=lambda val: f"ms{val}"),
        hyperparam("--tokens-per-sample", 8192, save_dir_key=lambda val: f"ts{val}"),
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
        hyperparam("--dropout", 0.1, save_dir_key=lambda val: f"dr{val}"),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"atdr{val}"),
        hyperparam("--relu-dropout", 0.0, save_dir_key=lambda val: f"actdr{val}"),
        hyperparam("--weight-decay", weight_decay, save_dir_key=lambda val: f"wd{val}"),
    ]

    # optimization settings
    grid += [
        hyperparam("--batch-size", 4, save_dir_key=lambda val: f"bsz{val}"),
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--adam-betas", "(0.9, 0.98)", save_dir_key=lambda val: "beta9999"),
        hyperparam("--adam-eps", adam_eps, save_dir_key=lambda val: f"eps{val}"),
        hyperparam("--clip-norm", 0.1, save_dir_key=lambda val: f"clip{val}"),
        hyperparam("--checkpoint-activations"),
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
        hyperparam("--reset-meters"),
        hyperparam("--reset-optimizer"),
        hyperparam("--reset-dataloader"),
    ]

    grid += [
        hyperparam("--share-all-embeddings"),
        hyperparam("--layernorm-embedding"),
        hyperparam("--share-decoder-input-output-embed"),
    ]

    # logging settings
    grid += [
        hyperparam("--skip-invalid-size-inputs-valid-test"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 10),
        hyperparam("--combine-val"),
        hyperparam("--save-interval-updates", 10000),
    ]
    grid += [
        hyperparam("--sample-ratio", 0.2, save_dir_key=lambda val: f"sample{val}"),
        hyperparam("--noise-density", 0.1, save_dir_key=lambda val: f"noise{val}"),
    ]

    if args.local:
        grid += [
            hyperparam("--log-format", "json"),
            hyperparam("--log-interval", 10),
        ]
    return grid


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
