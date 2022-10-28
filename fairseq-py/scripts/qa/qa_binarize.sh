#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.



TASK=$DATA_BIN


for SPLIT in train val test
do
  for LANG in source query target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json gpt2_bpe/encoder.json \
    --vocab-bpe gpt2_bpe/vocab.bpe \
    --inputs "$TASK/$SPLIT.$LANG" \
    --outputs "$TASK/$SPLIT.bpe.$LANG" \
    --workers 10 \
    --keep-empty; \
  done
done

for INPUT_TYPE in source query target
do
  fairseq-preprocess \
  --only-source \
  --trainpref "${TASK}/train.bpe.${INPUT_TYPE}" \
  --validpref "${TASK}/val.bpe.${INPUT_TYPE}" \
  --destdir "${TASK}/bin/$INPUT_TYPE" \
  --srcdict gpt2_bpe/dict.txt \
  --workers 60;
done


for INPUT_TYPE in source query
do
  fairseq-preprocess \
  --only-source \
  --testpref "${TASK}/test.bpe.${INPUT_TYPE}" \
  --destdir "${TASK}/bin/$INPUT_TYPE" \
  --srcdict gpt2_bpe/dict.txt \
  --workers 60;
done