#!/usr/bin/env python

import sweep
from sweep import hyperparam


def get_grid(args):
    grid = []

    total_num_udpates = 50000
    max_epoch = 50
    warmup_updates = 10
    num_data_loaders = 4
    arch = "bart_large"
    task = "summarization"
    criterion = "label_smoothed_cross_entropy"

    adam_eps = 1e-08
    weight_decay = 0.01

    grid += [
        hyperparam(
            "--restore-file",
            "/data/home/xwhan/fairseq-py/checkpoints/bart.large.cnn.extended4K-1K/model.pt",
        )
    ]

    # model settings
    grid += [
        hyperparam("--arch", arch, save_dir_key=lambda val: val),
        hyperparam("--task", task),
        hyperparam("--criterion", criterion),
        hyperparam("--max-source-positions", 4096),
        hyperparam("--max-target-positions", 1024),
        hyperparam("--source-lang", "source"),
        hyperparam("--target-lang", "target"),
        hyperparam("--truncate-source"),
        hyperparam("--label-smoothing", 0.1, save_dir_key=lambda val: f"ls{val}"),
        # hyperparam("--truncate-target"),
    ]

    grid += [
        hyperparam("--batch-size", 1, save_dir_key=lambda val: f"mt{val}"),
        hyperparam("--batch-size-valid", 1),
        hyperparam("--update-freq", [1,2], save_dir_key=lambda val: f"uf{val}"),
        hyperparam(
            "--max-epoch", max_epoch, save_dir_key=lambda val: f"me{val}"
        ),
        hyperparam("--required-batch-size-multiple", 1),
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
        hyperparam("--seed", 42, save_dir_key=lambda val: f"s{val}"),
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--adam-betas", "(0.9, 0.999)", save_dir_key=lambda val: "beta9999"),
        hyperparam("--total-num-update", total_num_udpates),
        hyperparam("--adam-eps", adam_eps, save_dir_key=lambda val: f"eps{val}"),
        hyperparam("--clip-norm", 0.1, save_dir_key=lambda val: f"clip{val}"),
    ]

    # lr scheduler
    grid += [
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam("--lr", [3e-5, 5e-6], save_dir_key=lambda val: f"lr{val}"),
        hyperparam(
            "--warmup-updates", warmup_updates, save_dir_key=lambda val: f"warm{val}"
        ),
    ]
    grid += [
        hyperparam("--fp16", save_dir_key=lambda val: "fp16"),
    ]

    # data loading settings
    grid += [
        hyperparam("--num-workers", num_data_loaders),
    ]

    # validation and checkpoint settings
    grid += [
        # hyperparam("--no-save"),
        hyperparam("--no-epoch-checkpoints"),
        hyperparam("--reset-meters"),
        hyperparam("--reset-optimizer"),
        hyperparam("--reset-dataloader"),
        hyperparam("--validate-interval-updates", 20),
        hyperparam("--best-checkpoint-metric", "rougel", save_dir_key=lambda val: f"cmetric{val}")
    ]

    grid += [
        hyperparam("--share-all-embeddings"),
        hyperparam("--layernorm-embedding"),
        hyperparam("--share-decoder-input-output-embed"),
        hyperparam("--find-unused-parameters")
    ]

    # logging settings
    grid += [
        hyperparam("--skip-invalid-size-inputs-valid-test"),
        hyperparam("--maximize-best-checkpoint-metric"),
        hyperparam("--log-format", "json"),
        hyperparam("--keep-last-epochs", 1),
        hyperparam("--log-interval", 10),
        hyperparam("--eval-rouge"),
        hyperparam("--eval-rouge-args", '{"beam": 4, "max_len_b": 300, "lenpen": 2.0, "no_repeat_ngram_size": 3}')
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
