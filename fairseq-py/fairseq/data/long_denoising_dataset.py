# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
import logging 
import random

from . import FairseqDataset, data_utils


logger = logging.getLogger(__name__)

def collate(
    samples,
    pad_idx,
    eos_idx,
    vocab,
    left_pad_source=False,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1
):
    assert input_feeding
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None, pad_to_multiple=1):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=None,  # use eos_idx of each sample instead of vocab.eos()
            left_pad=left_pad,
            move_eos_to_beginning=move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
        pad_to_multiple=pad_to_multiple
    )
    # sort by descending source length
    src_lengths = torch.LongTensor([s["source"].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None


    ### for model-based denoising ###
    masked_unfiltered = None
    if samples[0].get("masked_unfiltered", None) is not None:
        masked_unfiltered = merge(
            "masked_unfiltered",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["masked_unfiltered"]
            if pad_to_length is not None
            else None,
            pad_to_multiple=pad_to_multiple
        )
        masked_unfiltered = masked_unfiltered.index_select(0, sort_order)
    ### for model-based denoising ###

    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s["target"]) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s["source"]) for s in samples)

    batch = {
        "id": id,
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            'masked_unfiltered': masked_unfiltered,
        },
        "target": target,
        "nsentences": samples[0]["source"].size(0),
        "sort_order": sort_order,
    }

    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens

    return batch


class LongDenoisingDataset(FairseqDataset):
    """
    A wrapper around TokenBlockDataset for BART dataset.

    Args:
        dataset (TokenBlockDataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        mask_idx (int): dictionary index used for masked token
        mask_whole_words: only mask whole words. This should be a byte mask
            over vocab indices, indicating whether it is the beginning of a
            word. We will extend any mask to encompass the whole word.
        shuffle (bool, optional): shuffle the elements before batching.
          Default: ``True``
        seed: Seed for random number generator for reproducibility.
        args: argparse arguments.
    """

    def __init__(
        self,
        dataset,
        sizes,
        vocab,
        shuffle,
        seed,
        noise_density,
        mean_noise_span_length,
        sample_ratio=1,
        model_based=False,
        min_source_length=None,
        eos=None,
        truncate_target=False,
        pad_to_multiple=1,
        dynamic_span=False
    ):
        self.dataset = dataset
        self.sizes = sizes

        self.vocab = vocab
        self.shuffle = shuffle
        self.seed = seed

        self.min_source_length = min_source_length
        self.model_based = model_based
        self.truncate_target = truncate_target

        self.noise_density = noise_density # this is the initial masking ratio
        self.mean_noise_span_length = mean_noise_span_length
        self.sample_ratio = sample_ratio

        self.eos = eos if eos is not None else vocab.eos()
        self.sentinel_start = vocab.index("<sentinel_0>")

        self.epoch = 0
        self.pad_to_multiple = pad_to_multiple
        
        self.dynamic_span = dynamic_span
        if self.dynamic_span:
            avg_span_lens = [4, 8, 12]
            self.noisy_span_lens = [random.choice(avg_span_lens) for _ in range(len(sizes))]
        else:
            self.noisy_span_lens = None

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True  # only the noise changes, not item sizes

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def compute_lengths(self, orig_length, noisy_span_len):
        """
        calculate the source/target length

        # TODO the lengths calculation here is not exact
        """
        raw_length = orig_length - 2
        num_noise_tokens = int(round(raw_length * self.noise_density))
        num_noise_tokens *= self.sample_ratio

        num_nonnoise_tokens = raw_length - num_noise_tokens
        num_noise_spans = int(self.sample_ratio * round(num_noise_tokens / noisy_span_len))
        source_len = num_nonnoise_tokens + num_noise_spans + 2
        target_len = num_noise_tokens + num_noise_spans + 2

        # HACK the target lengths are handled within the model or via truncating
        if self.model_based or self.truncate_target:
            target_len = 1024

        return (int(source_len), int(target_len))

    def __getitem__(self, index):

        if self.dynamic_span:
            noisy_span_len = self.noisy_span_lens[index]
        else:
            noisy_span_len = self.mean_noise_span_length

        with data_utils.numpy_seed(self.seed, self.epoch, index):
            tokens = self.dataset[index]
            assert tokens[-1] == self.eos

            if tokens.size(0) <= 2:
                from random import randrange
                random_index = randrange(len(self.dataset))
                tokens = self.dataset[random_index]
            
            # @xwhan some incorrect processed samples?
            if self.model_based:
                source, masked_unfiltered = self.add_noise(tokens, noisy_span_len)
            else:
                source, target = self.add_noise(tokens, noisy_span_len)

        assert (source >= 0).all()
        assert (source[1:-1] >= 1).all()
        assert (source <= len(self.vocab)).all()
        assert source[0] == self.vocab.bos()
        assert source[-1] == self.eos
        return {
            "id": index,
            "source": source,
            "masked_unfiltered": None if not self.model_based else masked_unfiltered,
            "target": None if self.model_based else target,
        }

    def __len__(self):
        return len(self.dataset)

    def add_noise(self, source, noisy_span_len):
        length = source.size(0) - 2
        mask_indices = self.random_spans_noise_mask(length, noisy_span_len)
        labels_mask = ~mask_indices

        tokens = source[1:-1]

        if self.model_based:
            
            input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8)) 
            labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

            # masked inputs with <mask> only
            # MLM targets with masked positions as pad_idx
            masked = self.fill_input_ids(tokens, input_ids_sentinel, self.vocab.index("<mask>"))
            masked = torch.cat([source[:1], masked, source[-1:]])

            masked_target = torch.full(source.size(), self.vocab.pad()) 
            masked_target[1:-1] = self.fill_input_ids(tokens, labels_sentinel, self.vocab.pad())

            return masked, masked_target
            
        else:
            
            input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8), offset=self.sentinel_start - 1) 
            labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8), offset=self.sentinel_start - 1)

            input_ids = self.filter_input_ids(tokens, input_ids_sentinel)
            label_ids = self.filter_input_ids(tokens, labels_sentinel)

            if self.truncate_target:
                label_ids = label_ids[:1024-2]

            source = torch.cat([source[:1], input_ids, source[-1:]])
            target = torch.cat([source[:1], label_ids, source[-1:]])

            return source, target

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and squeeze consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        # batch_size = input_ids.shape[0]
        sentinel_ids = torch.tensor(sentinel_ids)
        input_ids_full = torch.where(sentinel_ids != 0, sentinel_ids, input_ids)
        input_ids = input_ids_full[input_ids_full > 0]
        return input_ids

    def fill_input_ids(self, input_ids, sentinel_ids, fill_idx):
        """
        set masked spans as <mask>
        """
        sentinel_ids = torch.tensor(sentinel_ids)
        masked_ids = torch.where(sentinel_ids != 0, fill_idx, input_ids)
        return masked_ids

    def create_sentinel_ids(self, mask_indices, offset=0):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[0] = mask_indices[0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, sentinel_ids + offset, 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def random_spans_noise_mask(self, length, noisy_span_len):
        orig_length = length
        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / noisy_span_len))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)
        return is_noise[:orig_length]


    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch of data
        """
        return collate(
            samples, self.vocab.pad(), self.eos, self.vocab, pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        if self.dynamic_span:
            noisy_span_len = self.noisy_span_lens[index]
        else:
            noisy_span_len = self.mean_noise_span_length
        return self.compute_lengths(self.sizes[index], noisy_span_len)[0]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.dynamic_span:
            noisy_span_len = self.noisy_span_lens[index]
        else:
            noisy_span_len = self.mean_noise_span_length
        return self.compute_lengths(self.sizes[index], noisy_span_len)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        
        if self.min_source_length:
            ignored = indices[self.sizes[indices] < self.min_source_length].tolist()
            indices = indices[self.sizes[indices] >= self.min_source_length]

            if len(ignored) > 0:
                logger.warning(
                    (
                        "{:,} samples have invalid sizes and will be skipped, "
                        "min_positions={}, first few sample ids={}"
                    ).format(len(ignored), self.min_source_length, ignored[:10])
                )

        return indices[np.argsort(self.sizes[indices], kind="mergesort")]


    def filter_indices_by_size(self, indices, max_sizes):
        """
        customized hacky funcion to reduce the time for building data iterator
        """
        if isinstance(max_sizes, float) or isinstance(max_sizes, int) or self.truncate_target: # if truncating elsewhere, then ignore the target limit
            
            if isinstance(max_sizes, tuple):
                max_sizes = max_sizes[0]

            if hasattr(self, "sizes") and isinstance(self.sizes, np.ndarray):
                ignored = indices[self.sizes[indices] > max_sizes].tolist()
                indices = indices[self.sizes[indices] <= max_sizes]
            elif (
                hasattr(self, "sizes")
                and isinstance(self.sizes, list)
                and len(self.sizes) == 1
            ):
                ignored = indices[self.sizes[0][indices] > max_sizes].tolist()
                indices = indices[self.sizes[0][indices] <= max_sizes]
            else:
                indices, ignored = data_utils._filter_by_size_dynamic(
                    indices, self.size, max_sizes
                )
        else:
            indices, ignored = data_utils._filter_by_size_dynamic(
                indices, self.size, max_sizes
            )
        return indices, ignored

    def prefetch(self, indices):
        self.src.prefetch(indices)
        self.tgt.prefetch(indices)

    @property
    def supports_prefetch(self):
        return (
            hasattr(self.src, "supports_prefetch")
            and self.src.supports_prefetch
            and hasattr(self.tgt, "supports_prefetch")
            and self.tgt.supports_prefetch
        )
