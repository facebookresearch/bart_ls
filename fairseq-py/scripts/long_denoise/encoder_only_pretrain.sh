#!/bin/bash

# export NCCL_DEBUG=INFO
# export FI_PROVIDER=efa
# export FI_OFI_RXR_RX_COPY_UNEXP=1
# export FI_OFI_RXR_RX_COPY_OOO=1
# export FI_EFA_MR_CACHE_ENABLE=1
# export FI_OFI_RXR_INLINE_MR_ENABLE=1
# export NCCL_TREE_THRESHOLD=0
# export NCCL_NET_SHARED_BUFFERS=0


DATA_DIR=/fsx/xwhan/data/pretrain_corpus/longformer_bookwiki_stories_realnes-bin
# DATA_DIR=/fsx/xwhan/data/pretrain_corpus/dialogue/mediasum/bin
# DATA_DIR=/fsx/xwhan/data/pretrain_corpus/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin/121219/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin/bookwiki

# python fb_sweep/sweep_long_pretrain.py -p blockwise_no_global_rerun  \
# -d ${DATA_DIR} \
# -g 8 -n 8 -t -1 --partition a100 --checkpoints-dir /checkpoints/xwhan/long_pretrain --resume-failed


# python fb_sweep/sweep_on_window_size.py -p block_512  \
# -d ${DATA_DIR} \
# -g 8 -n  -t -1 --partition a100 --checkpoints-dir /fsx/xwhan/checkpoints/long_pretrain --resume-failed

# python fb_sweep/sweep_long_pretrain.py -p performer_256 \
# -d ${DATA_DIR} \
# -g 8 -n 4 -t -1 --partition learnfair


## from roberta
python fb_sweep/sweep_from_roberta.py -p block512_lbsz_from_roberta_2  \
-d ${DATA_DIR} \
-g 8 -n 8 -t -1 --partition a100 --checkpoints-dir /fsx/xwhan/checkpoints/long_pretrain --resume-failed

# python fb_sweep/sweep_long_pretrain.py -p lsh_4 \
# -d ${DATA_DIR} \
# -g 1 -n 16 -t -1 --partition a100 --checkpoints-dir /checkpoints/xwhan/long_pretrain

# python train.py /fsx/xwhan/data/pretrain_corpus/dialogue/mediasum/bin --arch roberta_large  --task masked_lm --criterion masked_lm --update-freq 2 --max-update 10000 --required-batch-size-multiple 1 --dropout 0.0 --attention-dropout 0.0 --weight-decay 0.01 --optimizer adam --adam-eps 1e-08 --clip-norm 0.1 --lr-scheduler polynomial_decay --lr 3e-05 --total-num-update 5000 --warmup-updates 50 --no-epoch-checkpoints --reset-meters --reset-optimizer --skip-invalid-size-inputs-valid-test --log-format json --log-interval 100 --batch-size 1 --distributed-world-size 1 --max-positions 2048 --tokens-per-sample 2048 --combine-val --sample-break-mode complete --memory-efficient-fp16


# python train.py /fsx/xwhan/data/pretrain_corpus/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin/121219/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin/bookwiki --arch roberta_large  --task masked_lm --criterion masked_lm --update-freq 2 --max-update 10000 --required-batch-size-multiple 1 --dropout 0.0 --attention-dropout 0.0 --weight-decay 0.01 --optimizer adam --adam-eps 1e-08 --clip-norm 0.1 --lr-scheduler polynomial_decay --lr 3e-05 --total-num-update 5000 --warmup-updates 50 --no-epoch-checkpoints --reset-meters --reset-optimizer --skip-invalid-size-inputs-valid-test --log-format json --log-interval 100 --batch-size 1 --distributed-world-size 1 --max-positions 16384 --tokens-per-sample 16384 --combine-val --sample-break-mode complete --use-xformers --attention-name blocksparse_local --xformer-config '{"block_size": 512, "seq_len": 16384}' --memory-efficient-fp16

# --arch roberta_large --use-xformers --attention-name longshort cfru