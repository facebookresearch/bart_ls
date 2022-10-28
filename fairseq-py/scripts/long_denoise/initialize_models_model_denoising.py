# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from fairseq.tasks.denoising import DenoisingTask
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import os
import torch

from fairseq.tasks.masked_lm import MaskedLMTask


print('Loading RoBERTa weights...')
checkpoint_path = "/data/home/xwhan/fairseq-py/checkpoints/roberta.base"
dictionary = MaskedLMTask.load_dictionary(os.path.join(checkpoint_path, 'dict.txt'))
state = torch.load(os.path.join(checkpoint_path, 'model.pt'), map_location=torch.device('cpu'))
roberta_cfg = convert_namespace_to_omegaconf(state['args'])
task = MaskedLMTask(state['args'], dictionary)
roberta = task.build_model(roberta_cfg.model)
roberta.load_state_dict(state['model'], strict=True, model_cfg=roberta_cfg.model)

print('Loading BART weights...')
bart_path = "/data/home/xwhan/fairseq-py/checkpoints/bart.base"
dictionary = DenoisingTask.load_dictionary(os.path.join(bart_path, 'dict.txt'))
state = torch.load(os.path.join(bart_path, 'model.pt'), map_location=torch.device('cpu'))
bart_cfg = convert_namespace_to_omegaconf(state['args'])
task = DenoisingTask(state['args'], dictionary)
bart = task.build_model(bart_cfg.model)
bart.load_state_dict(state['model'], strict=True, model_cfg=bart_cfg.model)


from fairseq.tasks.model_based_denoising import ModelDenoisingTask

print(state['args'])
model_args = state['args']
model_args.arch = 'loco_base'
model_args.max_target_positions = 1024

# # 4k models, span 5
# model_args.xformer_config = '{"block_size": 1024, "max_seq_len": 4096}'
# model_args.max_source_positions = 1024 * 4
# model_args.mean_noise_span_length = 5
# model_args.noise_density = 0.2
# model_args.sample_ratio = 0.5
# model_args.tokens_per_sample = 1024 * 4

# 4k models, span 3
# model_args.xformer_config = '{"block_size": 1024, "max_seq_len": 4096}'
# model_args.max_source_positions = 1024 * 4
# model_args.mean_noise_span_length = 3
# model_args.noise_density = 0.1
# model_args.sample_ratio = 0.2
# model_args.tokens_per_sample = 1024 * 4

# # 8k models, span 3
# model_args.max_source_positions = 1024 * 8
# model_args.mean_noise_span_length = 3
# model_args.noise_density = 0.125
# model_args.sample_ratio = 0.3
# model_args.tokens_per_sample = 1024 * 8


# 16k models, span 3
model_args.max_source_positions = 1024 * 16
model_args.mean_noise_span_length = 3
model_args.noise_density = 1 / 16
model_args.sample_ratio = 0.2
model_args.tokens_per_sample = 1024 * 16
model_args.generator_layers = 6

# # 2k model 
# model_args.max_source_positions = 1024 * 8
# model_args.mean_noise_span_length = 5
# model_args.noise_density = 0.1
# model_args.sample_ratio = 0.2
# model_args.tokens_per_sample = 1024 * 8

# pooling layers
model_args.pooling_layers = 4

task = ModelDenoisingTask(model_args, dictionary)

long_cfg = convert_namespace_to_omegaconf(model_args)
long_model = task.build_model(long_cfg.model)


print("initializing the generator from roberta")
roberta = roberta.encoder

# position embeddings
pos_limit, _ = roberta.sentence_encoder.embed_positions.weight.shape
new_pos_limit, embed_dim = long_model.generator.sentence_encoder.embed_positions.weight.shape
new_pos_embed = roberta.sentence_encoder.embed_positions.weight.new_empty(new_pos_limit, embed_dim)
step = pos_limit - 2
for start in range(2, new_pos_limit, step):
    new_pos_embed[start:start+step] = roberta.sentence_encoder.embed_positions.weight[2:]
long_model.generator.sentence_encoder.embed_positions.weight.data = new_pos_embed

# vocab embedding matrix
vocab_size, _ = roberta.sentence_encoder.embed_tokens.weight.shape
new_vocab_size, embed_dim = long_model.generator.sentence_encoder.embed_tokens.weight.shape
print(f'roberta vocab size: {vocab_size}')
print(f'generator vocab size: {new_vocab_size}')
new_embed_tokens = roberta.sentence_encoder.embed_tokens.weight.new_empty(new_vocab_size, embed_dim)
new_embed_tokens[:vocab_size] = roberta.sentence_encoder.embed_tokens.weight
for idx in range(vocab_size, new_vocab_size):
    new_embed_tokens[idx] = roberta.sentence_encoder.embed_tokens.weight[-1] # initialize with <mask>
long_model.generator.sentence_encoder.embed_tokens.weight.data = new_embed_tokens

# layers and lm head
long_model.generator.lm_head.dense.load_state_dict(roberta.lm_head.dense.state_dict())
long_model.generator.lm_head.layer_norm.load_state_dict(roberta.lm_head.layer_norm.state_dict())
long_model.generator.sentence_encoder.layernorm_embedding.load_state_dict(roberta.sentence_encoder.layernorm_embedding.state_dict())

long_model.generator.sentence_encoder.layers.load_state_dict(roberta.sentence_encoder.layers[:model_args.generator_layers].state_dict())
print("done")

print("initializing the seq2seq model from bart")

##### encoder staff #####
## embed_tokens
vocab_size, _ = bart.encoder.embed_tokens.weight.shape
new_vocab_size, embed_dim = long_model.encoder.embed_tokens.weight.shape
print('old embedding matrix size from BART', vocab_size)
print('new embedding matrix size', new_vocab_size)
# how should we initialize these sentinel embeddings
new_embed_tokens = bart.encoder.embed_tokens.weight.new_empty(new_vocab_size, embed_dim)
new_embed_tokens[:vocab_size] = bart.encoder.embed_tokens.weight
for idx in range(vocab_size, new_vocab_size):
    new_embed_tokens[idx] = bart.encoder.embed_tokens.weight[-1] # initialize with <mask>
long_model.encoder.embed_tokens.weight.data = new_embed_tokens

## layernorm_embedding
long_model.encoder.layernorm_embedding.load_state_dict(bart.encoder.layernorm_embedding.state_dict())

## encoder layers
long_model.encoder.layers.load_state_dict(bart.encoder.layers.state_dict(), strict=False)

## embed positions
pos_limit, _ = bart.encoder.embed_positions.weight.shape
new_pos_limit, embed_dim = long_model.encoder.embed_positions.weight.shape
new_pos_embed = bart.encoder.embed_positions.weight.new_empty(new_pos_limit, embed_dim)
step = pos_limit - 2
for start in range(2, new_pos_limit, step):
    new_pos_embed[start:start+step] = bart.encoder.embed_positions.weight[2:]
long_model.encoder.embed_positions.weight.data = new_pos_embed

##### decoder staff #####
long_model.decoder.layernorm_embedding.load_state_dict(bart.decoder.layernorm_embedding.state_dict())
# 2. embed_positions, longer
long_model.decoder.embed_positions.load_state_dict(bart.decoder.embed_positions.state_dict())

# decoder attention layers
long_model.decoder.layers.load_state_dict(bart.decoder.layers.state_dict())

save_path = '/data/home/xwhan/fairseq-py/checkpoints/md.base.16k.pool4.span3.r6'
dictionary.save(os.path.join(save_path, 'dict.txt'))
state['args'] = model_args
state['model'] = long_model.state_dict()
if 'criterion' in state:
    del state['criterion']
state['extra_state'] = {"epoch": 0}
state['last_optimizer_state'] = None
torch.save(state, os.path.join(save_path, 'model.pt'))