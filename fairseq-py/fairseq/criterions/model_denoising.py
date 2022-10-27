# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class ModelDenoisingCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    mlm_loss_weight: float = field(
        default=1.0, 
        metadata={"help": "weight for mlm"}
    )

@register_criterion("model_based_denoising", dataclass=ModelDenoisingCriterionConfig)
class ModelBasedDenoisingCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg, mlm_loss_weight):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.mlm_loss_weight = mlm_loss_weight

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])

        seq2seq_loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        mlm_loss = None
        if len(net_output) == 4:
            mlm_loss = net_output[2]

        ntokens = (net_output[1] != self.padding_idx).sum()
        sample_size = ntokens

        logging_output = {
            "loss": seq2seq_loss,
            'mlm_loss': mlm_loss.data if mlm_loss else 0,
            "ntokens": ntokens,
            "nsentences": net_output[1].size(0),
            "sample_size": sample_size,
        }

        if mlm_loss is not None:
            loss = seq2seq_loss + mlm_loss * sample_size
        else:
            loss = seq2seq_loss

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        target = net_output[1].contiguous().view(-1)
        # target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        mlm_loss_sum = sum(log.get("mlm_loss", 0) for log in logging_outputs)

        metrics.log_scalar(
            "mlm_loss", mlm_loss_sum / len(logging_outputs), 1, round=3
        )

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
