# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python

import sweep
from sweep import hyperparam


def get_grid(args):

    model_size = 'large'
    fp16 = True # some models will run into issues with fp16
    max_update = 64000
    num_gpu = 64
    max_positions = 4096
    window_size_ablation = False

    # These parameterers can be fixed
    tokens_per_batch = 4096 * 64 * 2
    tokens_per_gpu = 4096 * 2 if model_size == 'large' and (not window_size_ablation) else 4096 * 4
    tokens_per_gpu = (tokens_per_gpu // 2) if not fp16 else tokens_per_gpu
    update_freq = tokens_per_batch // (tokens_per_gpu * num_gpu)
    warm_up_steps = 500
    # warm_up_steps = int(0.06 * max_update)

    return [
        hyperparam("--train-subset", "train" if not args.local else "valid"),
        hyperparam("--skip-invalid-size-inputs-valid-test"),
        hyperparam("--memory-efficient-fp16", save_dir_key=lambda val: "me_fp16"),
        hyperparam("--fast-stat-sync", save_dir_key=lambda _: "faststatsync"),
        hyperparam("--num-workers", 2),
        hyperparam("--task", "masked_lm"),
        hyperparam("--criterion", "masked_lm"),
        hyperparam("--max-positions", max_positions),
        hyperparam(
            "--sample-break-mode", "complete_doc", save_dir_key=lambda val: f"brk_{val}"
        ),
        hyperparam("--tokens-per-sample", max_positions, save_dir_key=lambda val: f"tps{val}"),
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--adam-betas", "(0.9, 0.98)", save_dir_key=lambda val: "b2_0.98"),
        hyperparam("--adam-eps", 1e-6, save_dir_key=lambda val: f"eps{val}"),
        hyperparam("--clip-norm", 0.0, save_dir_key=lambda val: f"cl{val}"),
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam("--lr", 3e-5, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--warmup-updates", warm_up_steps, save_dir_key=lambda val: f"wu{val}"), # use more updates for performer, 500 for other models
        hyperparam("--total-num-update", max_update),
        hyperparam("--dropout", 0.1, save_dir_key=lambda val: f"dr{val}"),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"atdr{val}"),
        hyperparam("--weight-decay", 0.01, save_dir_key=lambda val: f"wd{val}"),
        hyperparam("--batch-size", tokens_per_gpu // max_positions, save_dir_key=lambda val: f"ms{val}"),
        hyperparam("--update-freq", update_freq, save_dir_key=lambda val: f"uf{val}"),
        hyperparam("--max-update", max_update, save_dir_key=lambda val: f"mu{val}"),
        hyperparam("--seed", 42, save_dir_key=lambda val: f"s{val}"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 100),
        hyperparam("--combine-val"),
        hyperparam("--keep-last-epochs", 1),
        hyperparam("--save-interval-updates", 16000), # increased as it will save on epoch end
        hyperparam("--ddp-backend", "no_c10d"),
        hyperparam("--arch", f"roberta_{model_size}", save_dir_key=lambda val: val),
        hyperparam("--use-xformers"),
        hyperparam("--attention-name", "block"),
        hyperparam("--xformer-config", '{"window_size": 512}'),
        hyperparam("--restore-file", "/data/home/xwhan/fairseq-py/checkpoints/roberta.large.block-512/model.pt")
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
