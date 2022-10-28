# Adapting Pretrained Text-to-Text Models for Long Text Sequences

This repo contains code/checkpoints to reproduce the results of the paper: [Adapting Pretrained Text-to-Text Models for Long Text Sequences](https://arxiv.org/abs/2209.10052). We further pretrain the [BART](https://arxiv.org/abs/1910.13461) model for long sequence tasks, setting new state-of-the-art on abstract summarization of long texts (e.g., GovReport, BookSum, SummScreen, QMSum). Our implementation is based on custom forks of [fairseq](https://github.com/facebookresearch/fairseq) and [xformers](https://github.com/facebookresearch/xformers). You could use this repo to finetune on your own long-context tasks or implement efficienct long-context models while using the fast fairseq package.

## Environment Setup
Our model is developed using A100 GPUs and CUDA version 11.4, PyTorch 1.12.1. The exact result numbers might vary due to environment differences. 

* Install xformers and fairseq by running `pip install -e .` under their directory. Install apex following [https://github.com/NVIDIA/apex](https://github.com/NVIDIA/apex).

* Install Triton -- to suppress errors from xformers
```
pip install triton
```

* Install summarizaztion pyrouge and rouge_score
```
pip install -U  git+https://github.com/pltrdy/pyrouge
pip install rouge_score
```

## Summarization Performance

| Method      | GovReport (# Params) | BookSum-Chapters (# Params) | SummScreen-FD (# Params) | SummScreen-TVM (# Params)
------------------------------------| ----- | ----- | ----- | ----
|              | ROUGE-1/2 | ROUGE-1/2 | ROUGE-1/2 | ROUGE-1/2
Previous SOTA  | 61.0/28.8 (525M) | 38.3/9.2 (660M) | 36.8/9.2 (660M) | 51.0/14.7 (660M)
BART-LS (ours) 440M | **62.0/30.9** | **38.5/10.3** | **39.1/10.7** | **51.8/17.2**
 
## Model Checkpoints

Model Description                           | Download
--------------------------------------| ---
Pretrained Model   | [model_100k.pt](https://dl.fbaipublicfiles.com/lsbart/model_100k.pt)
Finetuned checkpoint on GovReport  | [model_gov.py](https://dl.fbaipublicfiles.com/lsbart/model_gov.pt)
Finetuned checkpoint SummScreen-fd | [model_fd.py](https://dl.fbaipublicfiles.com/lsbart/model_fd.pt)
Finetuned checkpoint on BookSum    | [model_book.py](https://dl.fbaipublicfiles.com/lsbart/model_book.pt)
Dictionary/vocabulary file    | [dict.txt](https://dl.fbaipublicfiles.com/lsbart/dict.txt)


## Code Structure

### Tasks
* Pretraining task: [fairseq-py/fairseq/tasks/long_denoising.py](https://github.com/facebookresearch/bart_ls/blob/main/fairseq-py/fairseq/tasks/long_denoising.py)
* Summarization task: [fairseq-py/fairseq/tasks/summarization.py](https://github.com/facebookresearch/bart_ls/blob/main/fairseq-py/fairseq/tasks/summarization.py)

### Architectures
* Pooling layers: [fairseq-py/fairseq/models/long_transformers/pooling_layers.py](https://github.com/facebookresearch/bart_ls/blob/main/fairseq-py/fairseq/models/long_transformers/pooling_layer.py)
* Block Attention: [xformers/xformers/components/attention/block_noglobal.py](https://github.com/facebookresearch/bart_ls/blob/main/xformers/xformers/components/attention/block_noglobal.py).
* Integration to fairseq's transformer architecture: [fairseq-py/fairseq/modules/multihead_attention.py](https://github.com/facebookresearch/bart_ls/blob/main/fairseq-py/fairseq/modules/multihead_attention.py#L181)

### Alternative Attention Implementations
Apart from the block attention implemented with native PyTorch operations, we also provides a faster version within xformers implemented with [Triton](https://github.com/openai/triton): [xformers/xformers/components/attention/blocksparse_local.py](https://github.com/facebookresearch/bart_ls/blob/main/xformers/xformers/components/attention/blocksparse_local.py). This implementation brings about 20-30% efficiency gains and slightly worse results. To enable this options, simply pass `--attention-name bs_local`. You can easy implement other architectures without worring about other transformer blocks.


## Instruction to finetuning the pretrained model

1. Prepare raw data. Organize you data as `{train|val|test}.{src|tgt}`, where each line corresponds to an example. 
2. Under fairseq-py/, binarize the data following `bash ./scripts/summarization/binarize.sh`. For query-based summarization, check `fairseq-py/scripts/summarization/qmsum_preprocess.sh`
3. The hyperparameters we used for each dataset can be found at [fairseq-py/fb_sweep/long_finetune/sweep_summ.py](https://github.com/facebookresearch/bart_ls/blob/main/fairseq-py/fb_sweep/long_finetune/sweep_summ.py). After downloading the checkpoints and put them under checkpoints/, use the following script to run finetuning:

```
bash scripts/summarization/ft_summ.sh
```

## 

## Using released summarization checkpoints

### Generating summarizes on summscreen
```
python scripts/summarization/long_generate.py \
            --model-dir ../checkpoints/model_fd.pt \
            --data-dir ${BINARIZED_DATA} \
            --save-dir ${SUMMARY_SAVE_DIR} \
            --split valid \
            --bsz 4

```
This script will print ROUGE numbers calculated by [rouge_score](https://pypi.org/project/rouge-score/), which is used by [Scrolls](https://www.scrolls-benchmark.com/leaderboard). In our paper, we reported the rouge scores using [files2rouge](https://github.com/pltrdy/files2rouge). Please follow their repo to install file2rouge and download standord-corenlp for tokenization.


## BibTeX
If you find the repo useful, please consider citing our paper:
```
@article{xiong2022adapting,
  title={Adapting Pretrained Text-to-Text Models for Long Text Sequences},
  author={Xiong, Wenhan and Gupta, Anchit and Toshniwal, Shubham and Mehdad, Yashar and Yih, Wen-tau},
  journal={arXiv preprint arXiv:2209.10052},
  year={2022}
}
```

## License
CC-BY-NC 4.0
