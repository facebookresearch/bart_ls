#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
1. remove empty line in each document 
2. add empty lines between documents
3. filter document with smaller length

"""


import json
from multiprocessing import Pool
import contextlib
import os
from typing import Counter
import sys


INPUT_FILE = "/fsx/xwhan/data/pretrain_corpus/realnews/realnews.jsonl"
ENCODER_JSON = "gpt2_bpe/encoder.json"
OUTPUT = "/fsx/xwhan/data/pretrain_corpus/realnews/realnews.bpe.filtered"
OUTPUT_RAW = "/fsx/xwhan/data/pretrain_corpus/realnews/realnews.txt"
MIN_LEN = 1200
VOCAB_JSON = "gpt2_bpe/vocab.bpe"

from fairseq.data.encoders.gpt2_bpe import get_encoder

class MultiprocessingEncoder(object):

    def initializer(self):
        global bpe
        bpe = get_encoder(ENCODER_JSON, VOCAB_JSON)

    def encode(self, line):
        global bpe
        ids = bpe.encode(line)
        return list(map(str, ids))

    def readitems(self, lines):
        raw_docs = []
        enc_docs = []
        for line in lines:
            line = line.strip()
            doc = json.loads(line.strip())['text']
            doc_text = os.linesep.join([s for s in doc.splitlines() if s])

            bpe_lines = [self.encode(s) for s in doc.splitlines() if s]
            doc_len = sum([len(l) for l in bpe_lines])
            if doc_len < MIN_LEN:
                continue
            raw_docs.append(doc_text)
            bpe_lines = os.linesep.join([" ".join(l) for l in bpe_lines])
            enc_docs.append(bpe_lines)
        return enc_docs, raw_docs


with contextlib.ExitStack() as stack:
    inputs = [
        stack.enter_context(open(INPUT_FILE, "r", encoding="utf-8"))
    ]
    outputs = [
        stack.enter_context(open(OUTPUT, "w", encoding="utf-8"))
    ]
    outputs_raw = [
        stack.enter_context(open(OUTPUT_RAW, "w", encoding="utf-8"))
    ]

    encoder = MultiprocessingEncoder()
    pool = Pool(60, initializer=encoder.initializer)
    encoded_lines = pool.imap(encoder.readitems, zip(*inputs), 100)

    stats = Counter()
    for i, (enc_lines, raw_lines) in enumerate(encoded_lines, start=1):
        for enc_line, output_h in zip(enc_lines, outputs):
            print(enc_line + "\n", file=output_h) # empty line between documents
        for raw_line, output_h in zip(raw_lines, outputs_raw):
            print(raw_line + "\n", file=output_h) # empty line between documents
        if i % 10000 == 0:
            print("processed {} lines".format(i), file=sys.stderr)

    for k, v in stats.most_common():
        print("[{}] filtered {} lines".format(k, v), file=sys.stderr)