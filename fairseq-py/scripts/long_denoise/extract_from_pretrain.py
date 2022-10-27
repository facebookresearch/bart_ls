
from fairseq.tasks.denoising import DenoisingTask
import torch
import os
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

import numpy as np

def extract_seq2seq(pretrained_path, model_path, dict_path, bart_path, pool=False):
    print('Extracting seq2seq model...')

    # load the pretrained model
    state = torch.load(model_path, map_location=torch.device('cpu'))
    dictionary = DenoisingTask.load_dictionary(dict_path)

    # build a 16k model from scratch
    if state['args'] is None:
        bart_state = torch.load(os.path.join(bart_path, 'model.pt'), map_location=torch.device('cpu'))
        state['args'] = bart_state['args']
    state['args'].max_source_positions = 1024 * 16
    state['args'].arch = 'bart_large'
    if pool:
        state['args'].pooling_layers = 4
    cfg = convert_namespace_to_omegaconf(state['args'])
    task = DenoisingTask(state['args'], dictionary)
    model = task.build_model(cfg.model)

    # # extend positions from pretrained models
    # pos_limit = state['model']['encoder.embed_positions.weight'].shape[0]
    # new_pos_limit, _ = model.encoder.embed_positions.weight.shape
    # pos_embeds = state['model']['encoder.embed_positions.weight'][2:].clone()
    # for _ in range((new_pos_limit - 2) // (pos_limit - 2) - 1):
    #     state['model']['encoder.embed_positions.weight'] = torch.cat([state['model']['encoder.embed_positions.weight'], pos_embeds], dim=0)

    model_state = {k: v for k, v in state['model'].items() if k.startswith('encoder.') or k.startswith('decoder.')}

    model.load_state_dict(model_state, strict=True)
    state['model'] = model.state_dict()
    if 'criterion' in state:
        del state['criterion']
    state['extra_state'] = {"epoch": 0}
    state['last_optimizer_state'] = None
    torch.save(state, os.path.join(pretrained_path, 'model_100k.pt'))
    dictionary.save(os.path.join(pretrained_path, 'dict.txt'))
    print('Done...')


# ###### pretrained model with pooled layers, block disjoint, 8k, span 3, 0.1 noise
# pretrain_path = "/checkpoints/xwhan/model_denoising/md_joint_pool.loco_large.pool4.ms8192.ts8192.mt1024.uf1.mu100000.brk_complete.dr0.1.atdr0.1.actdr0.0.wd0.01.bsz4.adam.beta9999.eps1e-06.clip0.1.s42.lr3e-05.warm500.memfp16.sample0.2.noise0.1.ngpu64/"
# model_path = pretrain_path + 'checkpoint_2_50000.pt'
# dict_path = "/data/home/xwhan/fairseq-py/checkpoints/local_large_v0/dict.txt"

###### pretrained model with pooled layers, block_sw, 4k, span 5
# pretrain_path = "/checkpoints/xwhan/model_denoising/md_joint_pool.loco_large.block_sw.pool4.ms4096.mt1024.uf1.mu100000.brk_complete.dr0.1.atdr0.1.actdr0.0.wd0.01.bsz4.adam.beta9999.eps1e-06.clip0.1.s42.lr3e-05.warm500.memfp16.sample0.5.noise0.2.spanlen5.ngpu64/"
# model_path = pretrain_path + 'checkpoint_2_100000.pt'
# dict_path = "/data/home/xwhan/fairseq-py/checkpoints/md.4k.pool4.span5/dict.txt"


##### pretrained model with pooled layers, block disjoint, 4k, span 5, 0.2 noise
# pretrain_path = "/checkpoints/xwhan/model_denoising/md_joint_pool.loco_large.bs_local.pool4.ms4096.mt1024.uf1.mu100000.brk_complete.dr0.1.atdr0.1.actdr0.0.wd0.01.bsz4.adam.beta9999.eps1e-06.clip0.1.s42.lr3e-05.warm500.memfp16.sample0.5.noise0.2.spanlen5.ngpu64/"
# model_path = pretrain_path + 'checkpoint_2_100000.pt'
# dict_path = "/data/home/xwhan/fairseq-py/checkpoints/md.4k.pool4.span5/dict.txt"
# bart_path = "/data/home/xwhan/fairseq-py/checkpoints/bart.large"

#### pretrained model with pooled layers, block disjoint, 4k, span 3, 0.1 noise, bsz 256
# pretrain_path = "/checkpoints/xwhan/model_denoising/md_joint_pool.loco_large.bs_local.pool4.ms4096.mt1024.uf1.mu100000.brk_complete.dr0.1.atdr0.1.actdr0.0.wd0.01.bsz8.adam.beta9999.eps1e-06.clip0.1.s42.lr3e-05.warm500.memfp16.sample0.2.noise0.1.spanlen3.ngpu32/"
# model_path = pretrain_path + 'checkpoint_2_100000.pt'
# dict_path = "/data/home/xwhan/fairseq-py/checkpoints/md.4k.pool4.span3/dict.txt"
# bart_path = "/data/home/xwhan/fairseq-py/checkpoints/bart.large"

# ###### pretrained model with pooled layers, block sw, 2k, span 3, 0.4 noise, bsz 1024
# pretrain_path = "/checkpoints/xwhan/model_denoising/md_joint_pool.loco_large.block_sw.pool4.ms2048.mt1024.uf2.mu100000.brk_complete.dr0.1.atdr0.1.actdr0.0.wd0.01.bsz8.adam.beta9999.eps1e-06.clip0.1.s42.lr3e-05.warm500.memfp16.sample0.8.noise0.4.spanlen3.ngpu64/"
# model_path = pretrain_path + 'checkpoint_2_50000.pt'
# dict_path = "/data/home/xwhan/fairseq-py/checkpoints/md.2k.pool4.span3/dict.txt"
# bart_path = "/data/home/xwhan/fairseq-py/checkpoints/bart.large"

# pretrain_path = '/data/home/xwhan/checkpoints/long_denoising/t5_baseline.bart_base.faststatsync.pool4.block_sw.ms8192.mt1024.uf2.mu100000.brk_complete_doc.dr0.1.atdr0.1.actdr0.0.wd0.01.bsz2.adam.beta9999.eps1e-06.clip0.1.s42.lr0.0001.warm500.memfp16.noise0.1.spanlen5.ngpu64/'
# model_path = pretrain_path + 'checkpoint_19_50000.pt'


pretrain_path = '/fsx/xwhan/checkpoints/long_denoising/t5_all_corpus.bart_large.faststatsync.pool4.block_noglobal.ms16384.mt1024.uf1.mu500000.brk_complete.dr0.1.atdr0.1.actdr0.0.wd0.01.bsz4.adam.eps1e-06.clip0.1.s42.lr0.0001.warm500.memfp16.noise0.0625.dynaspan.ngpu128/'
model_path = pretrain_path + 'checkpoint_37_100000.pt'

bart_path = "/data/home/xwhan/fairseq-py/checkpoints/bart.large"
dict_path = '/data/home/xwhan/fairseq-py/checkpoints/bart.large.block16k.pool.t5.span3/dict.txt'

extract_seq2seq(pretrain_path, model_path, dict_path, bart_path, pool=True)