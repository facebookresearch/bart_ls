#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

export CLASSPATH=/data/home/xwhan/repos/stanford-corenlp-4.2.2/stanford-corenlp-4.2.2.jar

cat $1 | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > /tmp/test.hypo.tokenized
cat $2 | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > /tmp/test.target.tokenized

files2rouge --ignore_empty_summary --ignore_empty_reference /tmp/test.target.tokenized /tmp/test.hypo.tokenized 