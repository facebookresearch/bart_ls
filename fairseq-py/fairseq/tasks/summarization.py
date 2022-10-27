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
import torch
import math

import numpy as np
from fairseq import metrics, utils
from fairseq.data import (
    FairseqDataset,
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
    ConcatSentencesDataset
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.tasks.translation import load_langpair_dataset
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from fairseq.data.encoders.gpt2_bpe import GPT2BPE, GPT2BPEConfig

# from rouge_score import rouge_scorer
from summ_eval import rouge_metric
import stanza

logger = logging.getLogger(__name__)

class MixedSentencesDataset(FairseqDataset):
    def __init__(self,
        eos, 
        blocksize=1024,
        pad=1,
        q_pad_len=0,
        sep=1,
        *datasets
        ):
        super().__init__()
        self.datasets = datasets
        self.blocksize = blocksize
        self.eos = eos
        self.pad = pad
        self.sep = sep
        self.q_pad_len = q_pad_len

        assert len(self.datasets) == 2
        assert all(
            len(ds) == len(datasets[0]) for ds in datasets
        ), "datasets must have the same length"

    def __getitem__(self, index):
        query = self.datasets[0][index]
        source = self.datasets[1][index]

        if self.q_pad_len != 0:
            # query should already be truncated here
            assert len(query) <= self.q_pad_len, (query, query.shape)
            pad_len = self.q_pad_len - len(query)
            query = torch.cat([query[:-1], query.new(pad_len).fill_(self.sep), query[-1:]])

        # each block ends with EOS
        source_bs = self.blocksize - len(query) -1
        blocks = []
        # for idx in range(0, len(source), source_bs):
        for idx in range(0, len(source), source_bs):
            chunk = source[idx:idx+source_bs]
            blocks.append(torch.cat([query, chunk, chunk.new([self.eos])]))
        return torch.cat(blocks)

    def __len__(self):
        return len(self.datasets[0])

    def collater(self, samples):
        return self.datasets[0].collater(samples)

    @property
    def sizes(self):
        return np.array([self.size(idx) for idx in range(len(self.datasets[0]))])

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        query_size, input_size = self.datasets[0].sizes[index], self.datasets[1].sizes[index]
        input_bs = self.blocksize - query_size - 1
        num_blocks = math.ceil(input_size / input_bs)
        return (query_size + 1) * num_blocks + input_size

    def ordered_indices(self):
        return self.datasets[0].ordered_indices()

    @property
    def supports_prefetch(self):
        return any(getattr(ds, "supports_prefetch", False) for ds in self.datasets)

    def prefetch(self, indices):
        for ds in self.datasets:
            if getattr(ds, "supports_prefetch", False):
                ds.prefetch(indices)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        for ds in self.datasets:
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)

def load_query_based_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    truncate_target=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    prepend_bos_src=None,
    qry='query', 
    max_query_positions=None,
    input_pattern='concat',
    blocksize=512, # only useful for "mixed" input patterns
    pad_q_len=0
):
    src_datasets = []
    tgt_datasets = []

    def split_exists(split, src, data_path):
        filename = os.path.join(data_path, "{}/{}".format(src, split))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        if not split_exists(split_k, src, data_path):
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            data_path + f'/{src}/{split_k}', src_dict, dataset_impl
        )

        assert blocksize > max_query_positions
        query_dataset = data_utils.load_indexed_dataset(data_path + f'/{qry}/{split_k}', src_dict, dataset_impl)

        if max_query_positions is not None:
            query_dataset = AppendTokenDataset(
                TruncateDataset(StripTokenDataset(query_dataset, src_dict.eos()), max_query_positions - 2), src_dict.eos()
                )

        query_dataset = PrependTokenDataset(query_dataset, src_dict.bos())

        if input_pattern == 'concat':
            src_dataset = ConcatSentencesDataset(query_dataset, src_dataset)
        elif input_pattern == 'mixed':
            # adding queries at multiple places
            src_dataset = MixedSentencesDataset(
                src_dict.eos(), 
                blocksize, 
                src_dict.pad(), 
                pad_q_len,
                src_dict.bos(),
                query_dataset, StripTokenDataset(src_dataset, src_dict.eos()))
        else:
            raise NotImplementedError

        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        # if split != 'test':
        tgt_dataset = data_utils.load_indexed_dataset(
            data_path + f'/{tgt}/{split_k}', tgt_dict, dataset_impl
        )

        if truncate_target:
            tgt_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(tgt_dataset, tgt_dict.eos()),
                    max_target_positions - 1,
                ),
                tgt_dict.eos(),
            )

        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)
        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0
    

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    elif prepend_bos_src is not None:
        logger.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )


@dataclass
class SummarizationConfig(FairseqDataclass):
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
    query_based: bool = field(default=False, metadata={"help": "query-based summarization"})
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

    # options for reporting rouge during validation
    eval_rouge: bool = field(
        default=False, metadata={"help": "evaluation with ROUGE scores"}
    )
    eval_rouge_args: Optional[str] = field(
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


@register_task("summarization", dataclass=SummarizationConfig)
class SummarizationTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: SummarizationConfig

    def __init__(self, cfg: SummarizationConfig, src_dict, tgt_dict):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        bpe_cfg = GPT2BPEConfig(
            gpt2_encoder_json='/datasets01/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin/121219/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin/encoder.json',
            gpt2_vocab_bpe='/datasets01/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin/121219/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin/vocab.bpe'
        )
        self.bpe = GPT2BPE(bpe_cfg)

        self.nlp = stanza.Pipeline(lang='en', processors='tokenize', verbose=False)

    @classmethod
    def setup_task(cls, cfg: SummarizationConfig, **kwargs):
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

        elif cfg.query_based:
            src_dict = cls.load_dictionary(
                os.path.join(paths[0], "{}/dict.txt".format(cfg.source_lang))
            )
            tgt_dict = cls.load_dictionary(
                os.path.join(paths[0], "{}/dict.txt".format(cfg.target_lang))
            )
        else:
            src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang))
        )
            tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang))
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

        if self.cfg.query_based:
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
        else:
            self.datasets[split] = load_langpair_dataset(
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
                shuffle=(split != "test" or split != 'valid'),
                pad_to_multiple=self.cfg.required_seq_len_multiple,
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
        if self.cfg.eval_rouge:
            detok_args = json.loads(self.cfg.eval_rouge_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.cfg.eval_rouge_detok, **detok_args)
            )

            gen_args = json.loads(self.cfg.eval_rouge_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.cfg.eval_rouge:
            rouge = self._inference_with_rouge(self.sequence_generator, sample, model)
            logging_output["rougel"] = rouge['rougel']
            logging_output["rouge1"] = rouge['rouge1']
            logging_output["rouge2"] = rouge['rouge2']
            logging_output["rouge_avg"] = (rouge['rouge1'] + rouge['rouge2']) / 2
            
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.cfg.eval_rouge:
            nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
            for metric in ['rouge1', 'rouge2', 'rougel', 'rouge_avg']:
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


    def _inference_with_rouge(self, generator, sample, model):

        def preprocess(text):
            doc = self.nlp(text)
            return '\n'.join(
                ' '.join(token.text for token in sentence.tokens)
                    for sentence in doc.sentences
            )

        # scorer = rouge_scorer.RougeScorer(rouge_types=['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scorer = rouge_metric.RougeMetric()
        rouge1 = rouge2 = rougel = 0.0

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(self.decode(utils.strip_pad(gen_out[i][0]["tokens"], self.tgt_dict.pad())))
            refs.append(
                self.decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                )
            )

            # empty data exists in gov report
            if len(hyps[-1].strip()) == 0 or len(refs[-1].strip()) == 0:
                hyps.pop()
                refs.pop()

        if self.cfg.eval_rouge_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])

        refs = [preprocess(ref) for ref in refs]
        hyps = [preprocess(hyp) for hyp in hyps]

        results = scorer.evaluate_batch(hyps, refs, aggregate=True)

        rouge1 += results['rouge']['rouge_1_f_score'] * 100 * len(refs)
        rouge2 += results['rouge']['rouge_2_f_score'] * 100 * len(refs)
        rougel += results['rouge']['rouge_l_f_score'] * 100 * len(refs)

        # for ref, pred in zip(refs, hyps):
        #     score = scorer.score(ref, pred)
        #     rouge1 += score['rouge1'].fmeasure
        #     rouge2 += score['rouge2'].fmeasure
        #     rougel += score['rougeL'].fmeasure

        return {'rougel': rougel, 'rouge1': rouge1, 'rouge2': rouge2}
