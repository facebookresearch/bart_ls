#!/bin/bash


# DATA_BIN=/fsx/xwhan/data/pretrain_corpus/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin/121219/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin/stories-mmap2-bin
# DATA_DIR="${DATA_BIN}/shard0"
# NUM_DATA_SHARDS=5
# for i in $(seq 1 $(($NUM_DATA_SHARDS-1)));
#   do
#     DATA_DIR="${DATA_DIR}:${DATA_BIN}/shard${i}";
#   done

DATA_DIR=/fsx/xwhan/data/pretrain_corpus/longformer_bookwiki_stories_realnes-bin
# DATA_DIR=/fsx/xwhan/data/pretrain_corpus/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin/121219/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin

python fb_sweep/sweep_long_efficiency.py -p final_speed_test \
-d ${DATA_DIR} \
-g 1 -n 1 -t 1 --partition a100 --checkpoints-dir /fsx/xwhan/checkpoints/long_efficiency --resume-failed

# python train.py /fsx/xwhan/data/pretrain_corpus/longformer_bookwiki_stories_realnes-bin --arch roberta_large  --task masked_lm --criterion masked_lm --update-freq 1 --max-update 200 --required-batch-size-multiple 1 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --optimizer adam --adam-eps 1e-08 --clip-norm 0.1 --lr-scheduler polynomial_decay --lr 3e-05 --total-num-update 1000 --warmup-updates 50 --no-epoch-checkpoints --reset-meters --reset-optimizer --skip-invalid-size-inputs-valid-test --log-format json --log-interval 10 --max-tokens 4096 --distributed-world-size 1 --max-positions 4096 --tokens-per-sample 4096 --combine-val --memory-efficient-fp16 --sample-break-mode complete_doc --use-xformers --attention-name lsh_reformer --xformer-config '{"num_hash": 4}'

# --arch roberta_large --use-xformers --attention-name longshort cfru