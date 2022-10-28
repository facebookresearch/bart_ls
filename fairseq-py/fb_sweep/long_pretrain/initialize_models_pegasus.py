# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from fairseq.models.bart import BARTModel

from fairseq.tasks.denoising import DenoisingTask
from fairseq.tasks.pegasus import PegasusTask
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import os
import torch

hub = BARTModel.from_pretrained('/data/home/xwhan/fairseq-py/checkpoints/bart.base', checkpoint_file='model.pt')
task = hub.task
bart = hub.model

model_args = hub.cfg.model

model_args.max_source_positions = 1024 * 16
model_args.max_target_positions = 1024
model_args.alibi = False
model_args.pooling_layers = 4

checkpoint_path = "/data/home/xwhan/fairseq-py/checkpoints/bart.base"
dictionary = DenoisingTask.load_dictionary(os.path.join(checkpoint_path, 'dict.txt'))
state = torch.load(os.path.join(checkpoint_path, 'model.pt'), map_location=torch.device('cpu'))

task = PegasusTask(model_args, dictionary)
long_cfg = convert_namespace_to_omegaconf(model_args)
long_model = task.build_model(long_cfg.model)

##### encoder staff #####

long_model.encoder.embed_tokens.load_state_dict(bart.encoder.embed_tokens.state_dict())
long_model.encoder.layernorm_embedding.load_state_dict(bart.encoder.layernorm_embedding.state_dict())

# 2. attention layers
long_model.encoder.layers.load_state_dict(bart.encoder.layers.state_dict(), strict=False)

# 3. embed_positions, longer
if not model_args.alibi:
    pos_limit, _ = bart.encoder.embed_positions.weight.shape
    new_pos_limit, embed_dim = long_model.encoder.embed_positions.weight.shape
    new_pos_embed = bart.encoder.embed_positions.weight.new_empty(new_pos_limit, embed_dim)
    step = pos_limit - 2
    for start in range(2, new_pos_limit, step):
        new_pos_embed[start:start+step] = bart.encoder.embed_positions.weight[2:]
    long_model.encoder.embed_positions.weight.data = new_pos_embed

##### decoder staff #####
long_model.decoder.load_state_dict(bart.decoder.state_dict())

save_path = '/data/home/xwhan/fairseq-py/checkpoints/bart.base.block16k.pool'
print(len(dictionary))
dictionary.save(os.path.join(save_path, 'dict.txt'))
state['args'] = model_args
state['model'] = long_model.state_dict()
if 'criterion' in state:
    del state['criterion']
state['extra_state'] = {"epoch": 0}
state['last_optimizer_state'] = None
torch.save(state, os.path.join(save_path, 'model.pt'))