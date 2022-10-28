#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


TASK=${YOUR_DATADIR}

for SPLIT in train val test
do
  for LANG in src tgt
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

fairseq-preprocess \
  --source-lang "src" \
  --target-lang "tgt" \
  --trainpref "${TASK}/train.bpe" \
  --validpref "${TASK}/val.bpe" \
  --destdir "${TASK}-bin/" \
  --workers 10 \
  --srcdict gpt2_bpe/dict.txt \
  --tgtdict gpt2_bpe/dict.txt;


fairseq-preprocess \
  --source-lang "src" \
  --only-source \
  --target-lang "tgt" \
  --testpref "${TASK}/test.bpe" \
  --destdir "${TASK}-bin/" \
  --workers 10 \
  --srcdict gpt2_bpe/dict.txt \
  --tgtdict gpt2_bpe/dict.txt;


# --testpref "${TASK}/test.bpe" \
# 