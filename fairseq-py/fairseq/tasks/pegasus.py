# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

from fairseq import utils
from fairseq.data import (
    AppendTokenDataset,
    Dictionary,
    PrependTokenDataset,
    StripTokenDataset,
    TokenBlockDataset,
    data_utils,
    PegasusDataset,
)
from fairseq.data.pegasus_dataset import PegasusDataset
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import LegacyFairseqTask, register_task
import numpy as np


logger = logging.getLogger(__name__)


@register_task("pegasus")
class PegasusTask(LegacyFairseqTask):
    """
    Denoising task for applying sequence to sequence denoising. (ie. BART)
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="path to data directory")
        parser.add_argument(
            "--tokens-per-sample",
            default=8192,
            type=int,
            help="max number of total tokens over all segments"
            " per sample for dataset",
        )
        parser.add_argument(
            "--sample-break-mode",
            default="complete_doc",
            type=str,
            help="mode for breaking sentence",
        )

        parser.add_argument(
            "--min-source-len",
            default=10,
            type=int,
        )
        parser.add_argument(
            "--max-target-len",
            default=1024,
            type=int,
        )
        parser.add_argument(
            "--mask-ratio",
            default=0.2,
            type=float,
            help="set the maximum sentence-masking ratio"
        )

        parser.add_argument(
            "--shorten-method",
            default="truncate",
            choices=["none", "truncate", "random_crop"],
            help="if not none, shorten sequences that exceed --tokens-per-sample",
        )
        parser.add_argument(
            "--shorten-data-split-list",
            default="",
            help="comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)',
        )
        parser.add_argument(
            "--truncate-target",
            action='store_true',
            help="comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)',
        )
        parser.add_argument(
            "--custom-dict",
            default=None,
            type=str
        )


    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed

        # add mask token
        self.mask_idx = self.dictionary.add_symbol("<sent_mask>")
        
    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task."""
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        if args.custom_dict:
            dictionary = Dictionary.load(args.custom_dict)
        else:
            dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.dictionary,
            self.args.dataset_impl,
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
            self.args.tokens_per_sample - 2,  # one less for <s> and one for </s>
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos(),
            break_mode=self.args.sample_break_mode,
            document_sep_len=0,
        )
        logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

        dataset = maybe_shorten_dataset(
            dataset,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.tokens_per_sample - 2,
            self.args.seed,
        )

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())
        dataset = AppendTokenDataset(dataset, self.source_dictionary.eos())

        self.datasets[split] = PegasusDataset(
            dataset,
            dataset.sizes,
            self.dictionary,
            shuffle=(split != "test" or split !='valid'),
            seed=self.seed,
            max_target_length=self.args.max_target_len,
            truncate_target=self.args.truncate_target,
            min_source_length=self.args.min_source_len,
            mask_ratio=self.args.mask_ratio,
            pad_to_multiple=self.args.required_seq_len_multiple
        )
        logger.info(
            "Split: {0}, Loaded {1} samples of denoising_dataset".format(
                split,
                len(self.datasets[split]),
            )
        )

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.dictionary
