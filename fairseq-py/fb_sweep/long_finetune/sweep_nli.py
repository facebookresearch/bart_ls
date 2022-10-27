#!/usr/bin/env python

from fb_sweep import sweep
from fb_sweep.sweep import hyperparam

"""


### r3f and ft on the scrolls



python fb_sweep/long_finetune/sweep_nli.py -p contract_r3f  \
-d /fsx/xwhan/data/scrolls/contract_nli/bin \
-g 8 -n 1 -t -1 --partition lowpri --checkpoints-dir /checkpoints/xwhan/contract --resume-failed --snapshot-code --baseline-model /fsx/xwhan/checkpoints/long_denoising/t5_all_corpus.bart_large.faststatsync.pool4.block_noglobal.ms16384.mt1024.uf1.mu500000.brk_complete.dr0.1.atdr0.1.actdr0.0.wd0.01.bsz4.adam.eps1e-06.clip0.1.s42.lr0.0001.warm500.memfp16.noise0.0625.dynaspan.ngpu128/model_100k.pt --time 1440 --local



"""

def get_grid(args):
    grid = []

    total_num_udpates = 8000
    warmup_updates = 200
    # warmup_updates = 800
    num_data_loaders = 8
    arch = "bart_large"
    # arch = "bart_prelayernorm"
    task = "qa"
    criterion = "label_smoothed_cross_entropy"
    adam_eps = 1e-08
    max_source_positions = 1024*16


    update_freq = 1

    max_q_pos = 35
    generate_args = '{"beam": 4, "max_len_b": 6, "lenpen": 3.0, "no_repeat_ngram_size": 3}'
    max_epochs = 60

    bsz = 4

    # better finetuning
    criterion = "label_smoothed_cross_entropy_r3f"
    # bsz = bsz//2
    # update_freq = update_freq*2
    grid += [
        hyperparam("--noise-type", ["uniform"], save_dir_key=lambda val: f"noise{val}"),
        hyperparam("--r3f-lambda", [0.01], save_dir_key=lambda val: f"r3f{val}"),
        hyperparam("--user-dir", "examples/rxf/rxf_src"),
        hyperparam("--ddp-backend", "no_c10d"),
        hyperparam("--reset-optimizer"),
    ]


    # which model to use
    grid += [
        hyperparam(
            "--custom-dict",

            '/data/home/xwhan/fairseq-py/checkpoints/bart.large.block16k.pool.t5.span3/dict.txt'

        )
    ]

    # model settings
    grid += [
        hyperparam("--arch", arch, save_dir_key=lambda val: val),
        hyperparam("--task", task),
        hyperparam("--required-seq-len-multiple", 1024),
        hyperparam("--criterion", criterion),
        hyperparam("--max-epoch", max_epochs, save_dir_key=lambda val: f"mep{val}"),
        hyperparam("--max-source-positions", max_source_positions, save_dir_key=lambda val: f"sl{val}"),
        hyperparam("--max-target-positions", 1024),
        hyperparam("--source-lang", "source"),
        hyperparam("--target-lang", "target"),
        hyperparam("--truncate-source"),
        hyperparam("--label-smoothing", 0.1, save_dir_key=lambda val: f"ls{val}"),
        hyperparam("--max-query-positions", max_q_pos), # narrativeqa 35
        hyperparam("--pad-query", 0, save_dir_key=lambda val: f"pad_q{val}"),
        hyperparam("--input-pattern", ['mixed'], save_dir_key=lambda val: f"ip{val}"),
        hyperparam("--use-xformers"),
        hyperparam("--pooling-layers", 4, save_dir_key=lambda val: f"pool{val}"),
        hyperparam("--attention-name", ['block_noglobal'], save_dir_key=lambda val: val),
        # hyperparam("--xformer-config", '{"num_global_tokens": 64}')
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
        hyperparam("--seed", [3, 42], save_dir_key=lambda val: f"s{val}"),
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--adam-betas", "(0.9, 0.999)"),
        hyperparam("--adam-eps", adam_eps, save_dir_key=lambda val: f"eps{val}"),
        hyperparam("--clip-norm", 0.1, save_dir_key=lambda val: f"clip{val}"),
        hyperparam("--checkpoint-activations"),
    ]

    # lr scheduler
    grid += [
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam("--lr", [5e-5, 3e-5], save_dir_key=lambda val: f"lr{val}"),
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

    metric = 'em'

    # validation and checkpoint settings
    grid += [
        hyperparam("--validate-interval", int(max_epochs // 5)),
        hyperparam("--no-epoch-checkpoints"),
        hyperparam("--validate-interval-updates", 200 if not args.local else 10),
        hyperparam("--best-checkpoint-metric", metric, save_dir_key=lambda val: f"cmetric{val}")
    ]

    # logging settings
    grid += [
        hyperparam("--skip-invalid-size-inputs-valid-test"),
        hyperparam("--maximize-best-checkpoint-metric"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 10),
        hyperparam("--eval-f1"),
        hyperparam("--generate-args", generate_args),
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
