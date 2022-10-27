from fairseq.models.bart import BARTModel

from fairseq.tasks.long_denoising import LongDenoisingTask
from fairseq.tasks.denoising import DenoisingTask
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import os
import torch

hub = BARTModel.from_pretrained('/data/home/xwhan/fairseq-py/checkpoints/bart.large', checkpoint_file='model.pt')
task = hub.task
bart = hub.model

model_args = hub.cfg.model

model_args.max_source_positions = 1024 * 16
model_args.max_target_positions = 1024
model_args.alibi = False
model_args.pooling_layers = 4

model_args.mean_noise_span_length = 3
model_args.noise_density = 0.0625

checkpoint_path = "/data/home/xwhan/fairseq-py/checkpoints/bart.large"
dictionary = DenoisingTask.load_dictionary(os.path.join(checkpoint_path, 'dict.txt'))
state = torch.load(os.path.join(checkpoint_path, 'model.pt'), map_location=torch.device('cpu'))

task = LongDenoisingTask(model_args, dictionary)
# task = DenoisingTask(model_args, dictionary)

long_cfg = convert_namespace_to_omegaconf(model_args)
long_model = task.build_model(long_cfg.model)


##### encoder staff #####
# 1. embed_tokens and layernorm_embedding
vocab_size, _ = bart.encoder.embed_tokens.weight.shape
new_vocab_size, embed_dim = long_model.encoder.embed_tokens.weight.shape
print('old embedding matrix size from BART', vocab_size)
print('new embedding matrix size', new_vocab_size)
new_embed_tokens = bart.encoder.embed_tokens.weight.new_empty(new_vocab_size, embed_dim)
new_embed_tokens[:vocab_size] = bart.encoder.embed_tokens.weight
for idx in range(vocab_size, new_vocab_size):
    new_embed_tokens[idx] = bart.encoder.embed_tokens.weight[-1]
long_model.encoder.embed_tokens.weight.data = new_embed_tokens
# 2. 
# long_model.encoder.embed_tokens.load_state_dict(bart.encoder.embed_tokens.state_dict())
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
long_model.decoder.layernorm_embedding.load_state_dict(bart.decoder.layernorm_embedding.state_dict())
if not model_args.alibi:
# 2. embed_positions, longer
    long_model.decoder.embed_positions.load_state_dict(bart.decoder.embed_positions.state_dict())

# decoder attention layers
long_model.decoder.layers.load_state_dict(bart.decoder.layers.state_dict())

save_path = '/data/home/xwhan/fairseq-py/checkpoints/bart.large.block16k.pool.t5.span3'
dictionary.save(os.path.join(save_path, 'dict.txt'))
state['args'] = model_args
state['model'] = long_model.state_dict()
if 'criterion' in state:
    del state['criterion']
state['extra_state'] = {"epoch": 0}
state['last_optimizer_state'] = None
torch.save(state, os.path.join(save_path, 'model.pt'))