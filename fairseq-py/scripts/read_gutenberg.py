#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from fairseq.data.encoders.gpt2_bpe import get_encoder

INPUT_FOLDER = "/fsx/xwhan/data/pretrain_corpus/gutenberg-dammit-files"
ENCODER_JSON = "gpt2_bpe/encoder.json"
VOCAB_JSON = "gpt2_bpe/vocab.bpe"


all_docs = []
for subdir, dirs, files in os.walk(INPUT_FOLDER):
    for file in files:
        if file.endswith('txt'):
            doc = open(os.path.join(subdir, file)).read().strip()
            all_docs.append(doc)
print(len(all_docs))