# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import itertools
import json
import logging
import os
from typing import Optional
from argparse import Namespace
from omegaconf import II

import numpy as np
from fairseq import metrics, utils
from fairseq.data import (
    LanguagePairDataset,
    data_utils,
    encoders,
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.tasks.summarization import load_query_based_dataset
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from fairseq.data.encoders.gpt2_bpe import GPT2BPE, GPT2BPEConfig

from rouge_score import rouge_scorer
import stanza

import re
import string
from collections import Counter
from multiprocessing import Pool

import nltk


logger = logging.getLogger(__name__)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)



@dataclass
class QAConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
            "in round-robin manner; however, valid and test data are always in the first directory "
            "to avoid the need for repeating them in all directories"
        },
    )
    source_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "source language",
            "argparse_alias": "-s",
        },
    )
    target_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "target language",
            "argparse_alias": "-t",
        },
    )
    load_alignments: bool = field(
        default=False, metadata={"help": "load the binarized alignments"}
    )
    left_pad_source: bool = field(
        default=False, metadata={"help": "pad the source on the left"}
    )
    left_pad_target: bool = field(
        default=False, metadata={"help": "pad the target on the left"}
    )
    max_source_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the target sequence"}
    )
    upsample_primary: int = field(
        default=-1, metadata={"help": "the amount of upsample primary dataset"}
    )
    truncate_source: bool = field(
        default=False, metadata={"help": "truncate source to max-source-positions"}
    )

    truncate_target: bool = field(
        default=False, metadata={"help": "truncate target to max-target-positions"}
    )

    # query based summarization for blockwise-encoder [Q+C1, Q+C2, ...]
    query_lang: Optional[str] = field(
        default='query',
        metadata={
        "help": "query suffix",
        "argparse_alias": "-s",
        },
    )
    max_query_positions: int = field(
        default=50, metadata={"help": "max number of tokens in the queries"}
    )
    pad_query: int = field(default=0, metadata={"help": "pad query to certain lengths for assigining global tokens"})
    input_pattern: str = field(
        default='concat', metadata={"help": "how to organize the query and input"}
    )
    block_size: int = field(
            default=1024, 
        )


    num_batch_buckets: int = field(
        default=0,
        metadata={
            "help": "if >0, then bucket source and target lengths into "
            "N buckets and pad accordingly; this is useful on TPUs to minimize the number of compilations"
        },
    )
    train_subset: str = II("dataset.train_subset")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    required_seq_len_multiple: int = II("dataset.required_seq_len_multiple")

    # qa metrics
    eval_f1: bool = field(
        default=False, metadata={"help": "token-level F1 scores as used by extractive QA"}
    )

    # options for reporting rouge during validation, i.e, used by narrativeQA
    eval_rouge: bool = field(
        default=False, metadata={"help": "evaluation with ROUGE scores"}
    )
    generate_args: Optional[str] = field(
        default="{}",
        metadata={
            "help": 'generation args for BLUE scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_rouge_detok: str = field(
        default="space",
        metadata={
            "help": "detokenize before computing rouge (e.g., 'moses'); required if using --eval-rouge; "
            "use 'space' to disable detokenization; see fairseq.data.encoders for other options"
        },
    )
    eval_rouge_detok_args: Optional[str] = field(
        default="{}",
        metadata={"help": "args for building the tokenizer, if needed, as JSON string"},
    )

    eval_rouge_remove_bpe: Optional[str] = field(
        default=None,
        metadata={
            "help": "remove BPE before computing rouge",
            "argparse_const": "@@ ",
        },
    )
    eval_rouge_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )

    custom_dict: Optional[str] = field(
        default=None,
        metadata={
            "help": "In case new dict.txt is introduced, compared of the dict files in the binarized data",
        },
    )


@register_task("qa", dataclass=QAConfig)
class QATask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: QAConfig

    def __init__(self, cfg: QAConfig, src_dict, tgt_dict):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        bpe_cfg = GPT2BPEConfig
        self.bpe = GPT2BPE(bpe_cfg)

        self.nlp = stanza.Pipeline(lang='en', processors='tokenize', verbose=False)

    @classmethod
    def setup_task(cls, cfg: QAConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(paths[0])
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        if cfg.custom_dict:
            src_dict = cls.load_dictionary(cfg.custom_dict)
            tgt_dict = cls.load_dictionary(cfg.custom_dict)

        else:
            src_dict = cls.load_dictionary(
                os.path.join(paths[0], "{}/dict.txt".format(cfg.source_lang))
            )
            tgt_dict = cls.load_dictionary(
                os.path.join(paths[0], "{}/dict.txt".format(cfg.target_lang))
            )

        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        return cls(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_query_based_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            load_alignments=self.cfg.load_alignments,
            truncate_source=self.cfg.truncate_source,
            truncate_target=self.cfg.truncate_target,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test" or split !='valid'),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
            qry=self.cfg.query_lang, 
            max_query_positions=self.cfg.max_query_positions,
            input_pattern=self.cfg.input_pattern,
            blocksize=self.cfg.block_size,
            pad_q_len=self.cfg.pad_query
            )


    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
            left_pad_source=self.cfg.left_pad_source,
            eos=self.src_dict.eos(),
            shuffle=False,
            pad_to_multiple=self.cfg.required_seq_len_multiple
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)
        detok_args = json.loads(self.cfg.eval_rouge_detok_args)
        self.tokenizer = encoders.build_tokenizer(
            Namespace(tokenizer=self.cfg.eval_rouge_detok, **detok_args)
        )

        gen_args = json.loads(self.cfg.generate_args)
        self.sequence_generator = self.build_generator(
            [model], Namespace(**gen_args)
        )
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        metric = self._inference_with_metrics(self.sequence_generator, sample, model)

        if self.cfg.eval_rouge:
            logging_output["rougel"] = metric['rougel']
            logging_output["rouge1"] = metric['rouge1']
            logging_output["rouge2"] = metric['rouge2']
            logging_output["rouge_avg"] = (metric['rouge1'] + metric['rouge2']) / 2
        if self.cfg.eval_f1:
            logging_output['em'] = metric['em']
            logging_output['f1'] = metric['f1']
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        if self.cfg.eval_rouge:
            for metric in ['rouge1', 'rouge2', 'rougel', 'rouge_avg']:
                sum_metric = sum(log.get(metric, 0) for log in logging_outputs)
                metrics.log_scalar(metric, sum_metric / nsentences)
        if self.cfg.eval_f1:
            for metric in ['em', 'f1']:
                sum_metric = sum(log.get(metric, 0) for log in logging_outputs)
                metrics.log_scalar(metric, sum_metric / nsentences)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict


    def decode(self, tokens):
        assert tokens.dim() == 1
        tokens = tokens.cpu().numpy()
        if tokens[0] == self.source_dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = tokens == self.source_dictionary.eos()
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [
            self.bpe.decode(self.source_dictionary.string(s)) for s in sentences
        ]
        if len(sentences) == 1:
            return sentences[0]
        return sentences


    def _inference_with_metrics(self, generator, sample, model):

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(self.decode(utils.strip_pad(gen_out[i][0]["tokens"], self.tgt_dict.pad())))
            refs.append(
                self.decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                )
            )

        if self.cfg.eval_rouge_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        
        metric = {}

        if self.cfg.eval_rouge:
            scorer = rouge_scorer.RougeScorer(rouge_types=['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
            rouge1 = rouge2 = rougel = 0.0

            hyps = ["\n".join(nltk.sent_tokenize(pred)) for pred in hyps]
            refs = ["\n".join(nltk.sent_tokenize(ref)) for ref in refs]
            for ref, pred in zip(refs, hyps):
                score = scorer.score(ref, pred)
                rouge1 += score['rouge1'].fmeasure
                rouge2 += score['rouge2'].fmeasure
                rougel += score['rougeL'].fmeasure

            metric.update({'rougel': rougel, 'rouge1': rouge1, 'rouge2': rouge2})
        
        if self.cfg.eval_f1:
            ems = f1s = 0.0
            for ref, hyp in zip(refs, hyps):
                ems += exact_match_score(hyp, ref)
                f1s += f1_score(hyp, ref)

            metric.update({'em': ems, 'f1': f1s})
            
        return metric


