# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import logging
import os
import math
from typing import Optional

from omegaconf import MISSING, II, OmegaConf

from fairseq.dataclass import ChoiceEnum
from fairseq.data.indexed_dataset import get_available_dataset_impl
import numpy as np
from fairseq import utils
from fairseq.data import (
    Dictionary,
    IdDataset,
    StripTokenDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    LongDenoisingDataset,
    PrependTokenDataset,
    AppendTokenDataset,
    RightPadDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
)
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

from .language_modeling import SAMPLE_BREAK_MODE_CHOICES, SHORTEN_METHOD_CHOICES


logger = logging.getLogger(__name__)


@dataclass
class ModelDenoisingConfig(FairseqDataclass):
    data: str = field(
        default=MISSING,
        metadata={
            "help": "colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner"
        },
    )
    sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
        default="none",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.'
        },
    )
    tokens_per_sample: int = field(
        default=8192,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )

    sample_ratio: float = field(
        default=0.5,
        metadata={"help": "masking ratio for the encoder-only model"},
    )
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    mask_whole_words: bool = field(
        default=False,
        metadata={"help": "mask whole words; you may also want to set --bpe"},
    )
    mask_multiple_length: int = field(
        default=1,
        metadata={"help": "repeat the mask indices multiple times"},
    )
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="truncate",
        metadata={
            "help": "if not none, shorten sequences that exceed --tokens-per-sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    required_seq_len_multiple: int = II("dataset.required_seq_len_multiple")

    truncate_target: bool = field(
        default=False, metadata={"help": "truncate target to max-target-positions"}
    )

    min_source_len: int = field(
        default=10,
        metadata={"help": "avg of masked span lengths"},
    )

    # T5-style random-span denoising
    noise_density: float = field(
        default=0.15,
        metadata={"help": "noise density after filtering"},
    )
    mean_noise_span_length: int = field(
        default=3,
        metadata={"help": "avg of masked span lengths"},
    )

    dynamic_span_len: bool = field(
        default=False
    )

    seed: int = II("common.seed")

    custom_dict: Optional[str] = field(
        default=None,
        metadata={
            "help": "In case new dict.txt is introduced, compared of the dict files in the binarized data",
        },
    )


@register_task("model_based_denoising", dataclass=ModelDenoisingConfig)
class ModelDenoisingTask(FairseqTask):

    cfg: ModelDenoisingConfig

    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    def __init__(self, cfg: ModelDenoisingConfig, dictionary):
        super().__init__(cfg)
        self.dictionary = dictionary

        # add mask token
        self.mask_idx = dictionary.add_symbol("<mask>")

        # add sentinel tokens
        max_num_noise_tokens = int(np.round((cfg.tokens_per_sample - 2) * (cfg.noise_density / cfg.sample_ratio)))
        max_num_noise_spans = int(np.round(max_num_noise_tokens / cfg.mean_noise_span_length))
        max_num_noise_spans = math.ceil(max_num_noise_spans * cfg.sample_ratio)

        buffer = 1 # FIXME is this necessary
        logger.info(f"Adding {max_num_noise_spans + buffer} new sentinel tokens")
        for sentinel_id in range(max_num_noise_spans + buffer):
            self.dictionary.add_symbol(f"<sentinel_{sentinel_id}>")

    @classmethod
    def setup_task(cls, cfg: ModelDenoisingConfig, **kwargs):
        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        if cfg.custom_dict:
            dictionary = Dictionary.load(cfg.custom_dict)
        else:
            dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(cfg, dictionary)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        # each instance ends with the eos token
        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.source_dictionary,
            self.cfg.dataset_impl,
            combine=combine,
        )
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        dataset = StripTokenDataset(dataset, self.dictionary.eos())

        # create continuous blocks of tokens
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes - 1,
            self.cfg.tokens_per_sample - 2,  # one less for <s>
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode=self.cfg.sample_break_mode,
            document_sep_len=0,
        )
        logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

        dataset = maybe_shorten_dataset(
            dataset,
            split,
            self.cfg.shorten_data_split_list,
            self.cfg.shorten_method,
            self.cfg.tokens_per_sample - 2,
            self.cfg.seed,
        )

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())
        dataset = AppendTokenDataset(dataset, self.source_dictionary.eos())

        self.datasets[split] = LongDenoisingDataset(
            dataset,
            dataset.sizes,
            self.dictionary,
            shuffle=(split != "test" or split !='valid'),
            seed=self.cfg.seed,
            model_based=True,
            noise_density=self.cfg.noise_density / self.cfg.sample_ratio, # initial masking ratio
            mean_noise_span_length=self.cfg.mean_noise_span_length,
            truncate_target=self.cfg.truncate_target,
            sample_ratio=self.cfg.sample_ratio,
            min_source_length=self.cfg.min_source_len,
            pad_to_multiple=self.cfg.required_seq_len_multiple,
            dynamic_span=self.cfg.dynamic_span_len
        )
        logger.info(
            "Split: {0}, Loaded {1} samples of denoising_dataset".format(
                split,
                len(self.datasets[split]),
            )
        )

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
