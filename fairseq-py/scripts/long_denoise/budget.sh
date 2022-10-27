#!/bin/bash


# DATA_DIR=/fsx/xwhan/data/pretrain_corpus/dialogue/bin
DATA_DIR=/fsx/xwhan/data/pretrain_corpus/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin/121219/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin/bookwiki


python fb_sweep/sweep_t5.py -p t5_budget \
-d ${DATA_DIR} \
-g 8 -n 1 -t 1 --partition a100 --checkpoints-dir /fsx/xwhan/checkpoints/t5_budget --resume-failed --local