# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
import logging
# from fairseq.data.encoders.gpt2_bpe import GPT2BPE, GPT2BPEConfig
from fairseq.data.denoising_dataset import collate
import math
from collections import Counter

from . import FairseqDataset, data_utils

logger = logging.getLogger(__name__)


def _score_ngrams(target_ngrams, prediction_ngrams):
    """Compute n-gram overlap scores

    each ngram is counted once as in Pegasus paper
    """
    target_ngrams = set(target_ngrams.keys())
    prediction_ngrams = set(prediction_ngrams.keys())
    intersection_ngrams_count = len(target_ngrams.intersection(prediction_ngrams))

    target_ngrams_count = len(target_ngrams)
    prediction_ngrams_count = len(prediction_ngrams)

    precision = intersection_ngrams_count / max(prediction_ngrams_count, 1)
    recall = intersection_ngrams_count / max(target_ngrams_count, 1)

    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    else:
        return 0.0



class PegasusDataset(FairseqDataset):
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
        max_target_length=1024,
        min_source_length=None,
        eos=None,
        truncate_target=False,
        mask_ratio=0.15,
        pad_to_multiple=1,
    ):
        self.dataset = dataset
        self.sizes = sizes

        self.vocab = vocab
        self.shuffle = shuffle
        self.seed = seed

        self.min_source_length = min_source_length
        self.truncate_target = truncate_target
        self.max_target_length = max_target_length
        self.mask_ratio = mask_ratio

        self.eos = eos if eos is not None else vocab.eos()

        self.full_stop_index = self.vocab.index("13")

        self.sent_mask_idx = self.vocab.index("<sent_mask>")

        # bpe_cfg = GPT2BPEConfig
        # self.bpe = GPT2BPE(bpe_cfg)
        # breakpoint()
        # partial_stops = ';!,' # TODO other punctuations?
        # partial_stops_bpe = [self.bpe.encode(c) for c in partial_stops]
        # breakpoint()
        self.partial_stop_indices = [self.vocab.index(c) for c in ['26', '11', '0']]

        self.epoch = 0
        self.pad_to_multiple = pad_to_multiple

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True  # only the noise changes, not item sizes

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch


    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            tokens = self.dataset[index]
            assert tokens[-1] == self.eos
            
            source, target = self.search_primaries(tokens)

        assert (source >= 0).all()
        assert (source[1:-1] >= 1).all()
        assert (source <= len(self.vocab)).all()
        assert source[0] == self.vocab.bos()
        assert source[-1] == self.eos
        return {
            "id": index,
            "source": source,
            "target": target,
        }

    def __len__(self):
        return len(self.dataset)

    def search_primaries(self, source):

        tokens = source[1:-1]

        full_stops = tokens == self.full_stop_index
        full_stops[-1] = 1

        sentence_ends = (full_stops[1:] * ~full_stops[:-1]).nonzero(as_tuple=False) + 2
        num_sentences = sentence_ends.size(0)

        if num_sentences < 2:
            # backoff to more punctuations
            full_stops = torch.zeros_like(tokens)
            for idx in range(len(tokens)):
                full_stops[idx] = tokens[idx] in self.partial_stop_indices
            full_stops[-2] = 1
            sentence_ends = (full_stops[1:] * ~full_stops[:-1]).nonzero(as_tuple=False) + 2
            num_sentences = sentence_ends.size(0)

        if num_sentences < 2:
            # backoff to simple denoising
            return self.random_delete(source)

        sentence_scores = []
        all_gram_counter = Counter(self.vocab.string(tokens).split())
        for i in range(num_sentences):
            start = sentence_ends[i - 1] if i > 0 else 1
            end = sentence_ends[i]
            sentence = tokens[start: end]
            score = self.score_sentence(sentence, all_gram_counter)
            sentence_scores.append(
                (score, (start, end))
            )
        sentence_scores.sort(reverse=True, key=lambda x:x[0])
        
        top_m = math.ceil(len(sentence_scores)*self.mask_ratio)
        tgt_len = 0
        tgt_spans = []
        for idx in range(top_m):
            s, e = sentence_scores[idx][1]
            tgt_len += e - s + 1
            tgt_spans.append((s, e))
            if tgt_len >= self.max_target_length - 2:
                break
        tgt_spans.sort(key=lambda x:x[0])

        last_end = 0
        src_tokens, tgt_tokens = [], []
        for span in tgt_spans:
            src_tokens.append(tokens[last_end:span[0]]) # TODO add mask
            src_tokens.append(torch.tensor([self.sent_mask_idx]))
            assert len(tokens[span[0]:span[1]]) > 0
            tgt_tokens.append(tokens[span[0]:span[1]])
            last_end = span[1]

        src_seq = torch.cat(src_tokens)
        tgt_seq = torch.cat(tgt_tokens)
        
        input = torch.cat([source[:1], src_seq, source[-1:]], dim=-1)
        tgt_seq = tgt_seq[:self.max_target_length - 2]
        target = torch.cat([source[:1], tgt_seq, source[-1:]], dim=-1)
        
        return input, target

    def score_sentence(self, sent, all_gram_counter):
        # str_pred = self.bpe.decode(self.vocab.string(pred))
        # str_ref = self.bpe.decode(self.vocab.string(ref))
        str_sent = self.vocab.string(sent)
        sent_counter = Counter(str_sent.split())
        rest_counter = all_gram_counter - sent_counter

        return _score_ngrams(rest_counter, sent_counter)

    def random_delete(self, source):
        tokens = source[1:-1]
        input, output = self.random_span(tokens)
        input = torch.cat([source[:1],input, source[-1:]])
        target = torch.cat([source[:1], output, source[-1:]])
        return input, target

    def random_span(self, tokens):
        length = len(tokens)

        num_noise_tokens = int(np.round(length * self.mask_ratio))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_remain_tokens = length - num_noise_tokens
        separate = torch.randint(0, num_remain_tokens + 1, (1,))

        output = tokens[separate:separate+num_noise_tokens]
        input = torch.cat([tokens[:separate], torch.tensor([self.sent_mask_idx]), tokens[separate+num_noise_tokens:]])

        return input, output

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
        return self.sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.compute_lengths(self.sizes[index])

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
