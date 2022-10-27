#!/bin/bash

# mkdir -p gpt2_bpe
# wget -O gpt2_bpe/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
# wget -O gpt2_bpe/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe

# DATA_DIR=/fsx/xwhan/data/pretrain_corpus/dialogue/mediasum

# for SPLIT in train valid; do \
#     python -m examples.roberta.multiprocessing_bpe_encoder \
#         --encoder-json gpt2_bpe/encoder.json \
#         --vocab-bpe gpt2_bpe/vocab.bpe \
#         --inputs ${DATA_DIR}/${SPLIT}.txt \
#         --outputs ${DATA_DIR}/${SPLIT}.bpe \
#         --keep-empty \
#         --workers 60; \
# done


DATA_DIR=/fsx/xwhan/data/pretrain_corpus/c4


# for SPLIT in {0..9}; do \
#     python -m examples.roberta.multiprocessing_bpe_encoder \
#         --encoder-json gpt2_bpe/encoder.json \
#         --vocab-bpe gpt2_bpe/vocab.bpe \
#         --inputs ${DATA_DIR}/train_${SPLIT}.txt \
#         --outputs ${DATA_DIR}/bpe/train_${SPLIT}.bpe \
#         --keep-empty \
#         --workers 60; \
# done



# wget -O gpt2_bpe/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt


# for SPLIT in {1..9}; do \
#     fairseq-preprocess \
#         --only-source \
#         --srcdict gpt2_bpe/dict.txt \
#         --trainpref ${DATA_DIR}/bpe/train_${SPLIT}.bpe \
#         --destdir ${DATA_DIR}/bin/train${SPLIT} \
#         --workers 60
# done



fairseq-preprocess \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --validpref ${DATA_DIR}/bpe/valid.bpe \
    --destdir ${DATA_DIR}/bin/valid \
    --workers 60