#!/usr/bin/env python

from fb_sweep import sweep
from fb_sweep.sweep import hyperparam


def get_grid(args):

    model_size = 'large'
    fp16 = True # some models will run into issues with fp16
    max_update = 100000
    num_gpu = 64
    max_positions = 4096
    lr = 6e-4
    window_size = 512

    # These parameterers can be fixed
    tokens_per_batch = 2048 * 512
    tokens_per_gpu = 4096 * 2 # TODO hack
    tokens_per_gpu = (tokens_per_gpu // 2) if not fp16 else tokens_per_gpu
    update_freq = int(tokens_per_batch / (tokens_per_gpu * num_gpu))
    warm_up_steps = int(0.06 * max_update)

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
            "--sample-break-mode", ["complete"], save_dir_key=lambda val: f"brk_{val}"
        ),
        hyperparam("--tokens-per-sample", max_positions, save_dir_key=lambda val: f"tps{val}"),
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--adam-betas", "(0.9, 0.98)", save_dir_key=lambda val: "b2_0.98"),
        hyperparam("--adam-eps", 1e-6, save_dir_key=lambda val: f"eps{val}"),
        hyperparam("--clip-norm", 0.0, save_dir_key=lambda val: f"cl{val}"),
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam("--lr", lr, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--warmup-updates", warm_up_steps, save_dir_key=lambda val: f"wu{val}"), # use more updates for performer, 500 for other models
        hyperparam("--total-num-update", max_update),
        hyperparam("--dropout", 0.1, save_dir_key=lambda val: f"dr{val}"),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"atdr{val}"),
        hyperparam("--weight-decay", 0.01, save_dir_key=lambda val: f"wd{val}"),
        hyperparam("--batch-size", int(tokens_per_gpu // max_positions), save_dir_key=lambda val: f"ms{val}"),
        hyperparam("--update-freq", update_freq, save_dir_key=lambda val: f"uf{val}"),
        hyperparam("--max-update", max_update, save_dir_key=lambda val: f"mu{val}"),
        hyperparam("--seed", 42, save_dir_key=lambda val: f"s{val}"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 100),
        hyperparam("--combine-val"),
        hyperparam("--keep-last-epochs", 1),
        hyperparam("--save-interval-updates", 20000), # increased as it will save on epoch end
        hyperparam("--encoder-normalize-before", save_dir_key=lambda val: "preln"),
        hyperparam("--ddp-backend", "no_c10d"),
        hyperparam("--arch", f"roberta_{model_size}", save_dir_key=lambda val: val),
        hyperparam("--use-xformers"),
        hyperparam("--attention-name", "block"),
        hyperparam("--xformer-config", '{"window_size": 512}'),
        hyperparam("--restore-file", "/fsx/xwhan/checkpoints/long_pretrain/block_512.me_fp16.faststatsync.brk_complete.tps4096.adam.b2_0.98.eps1e-06.cl0.0.lr0.0006.wu6000.dr0.1.atdr0.1.wd0.01.ms2.uf2.mu100000.s42.preln.roberta_large.ngpu64/checkpoint_5_80000.pt"),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
