#!/usr/bin/env python

from fb_sweep import sweep
from fb_sweep.sweep import hyperparam


"""
python fb_sweep/xformers_analysis/sweep_long_pretrain.py -p books \
-d /fsx/xwhan/data/pretrain_corpus/bookwiki_stories_realnews-bin/stories-mmap2-bin/shard0 \
-g 1 -n 1 -t -1 --partition a100 --checkpoints-dir /checkpoints/xwhan/long_pretrain --tensorboard-logdir /checkpoints/xwhan/long_context_analysis/tb --no-wandb --local

"""


def get_grid(args):

    model_size = 'large'
    fp16 = True # some models will run into issues with fp16
    max_update = 100000
    num_gpu = 32
    max_positions = 4096
    lr = 6e-4
    window_size_ablation = False

    # These parameterers can be fixed
    tokens_per_batch = 2048 * 512
    tokens_per_gpu = 4096 * 2 if model_size == 'large' and (not window_size_ablation) else 4096 * 4
    tokens_per_gpu = (tokens_per_gpu // 2) if not fp16 else tokens_per_gpu
    update_freq = tokens_per_batch // (tokens_per_gpu * num_gpu)
    warm_up_steps = int(0.06 * max_update)

    return [
        hyperparam("--train-subset", "train" if not args.local else "valid"),
        hyperparam("--skip-invalid-size-inputs-valid-test"),
        # hyperparam("--memory-efficient-fp16", save_dir_key=lambda val: "me_fp16"),
        hyperparam("--fast-stat-sync", save_dir_key=lambda _: "faststatsync"),
        hyperparam("--num-workers", 4),
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
        hyperparam("--batch-size", tokens_per_gpu // max_positions, save_dir_key=lambda val: f"ms{val}"),
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
        # hyperparam("--s4", save_dir_key=lambda val: 's4'),
        # hyperparam("--use-xformers"),
        # hyperparam("--attention-name", "block_noglobal", save_dir_key=lambda val: val),
        # hyperparam("--xformer-config", '{"block_size": 512}'),
        # hyperparam("--reset-optimizer"),
        # hyperparam("--restore-file", "/checkpoints/xwhan/long_pretrain/nystrom_256_bwsr_rerun.me_fp16.faststatsync.brk_complete.tps4096.adam.b2_0.98.eps1e-06.cl0.0.lr0.0006.wu6000.dr0.1.atdr0.1.wd0.01.ms2.uf4.mu100000.s42.preln.roberta_large.ngpu32/checkpoint_last.pt")
        # hyperparam("--xformer-config", '{"window_size": 64}'),
        # hyperparam("--memory-efficient-fp16", save_dir_key=lambda val: "me_fp16"),
        # hyperparam("--arch", f"linformer_roberta_{model_size}", save_dir_key=lambda val: val),
        # hyperparam("--user-dir", "examples/linformer/linformer_src"),
        # hyperparam("--compressed", [8], save_dir_key=lambda val: f"compress{val}"),
       
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
