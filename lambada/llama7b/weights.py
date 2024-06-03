import torch
import struct
import numpy as np
from tqdm import tqdm
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

# model_path = 'openlm-research/open_llama_7b'
# model_path = 'meta-llama/Llama-2-13b-chat-hf'
model_path = 'meta-llama/Llama-2-7b-chat-hf'

model = AutoModelForCausalLM.from_pretrained(
    model_path)
sd = model.state_dict()

for k in sd.keys():
    sd[k] = sd[k].cpu()

print(sd.keys())

f = open('weights.dat', 'wb')
# f = open('meta_llama2_13b.dat', 'wb')
# f = open(sys.argv[1], 'wb')

def dumpvec(x):
    # print(x.shape)
    assert(len(x.shape) == 1)
    f.write(x.numpy().tobytes())

def dumpmat(w):
    # print(w.shape)
    assert(len(w.shape) == 2)
    f.write(w.numpy().tobytes())

for i in tqdm(range(32)):
# for i in tqdm(range(40)):

    dumpvec(sd['model.layers.%d.input_layernorm.weight' % i])

    # c_attn_w = np.concatenate([sd['transformer.h.%d.attn.attention.q_proj.weight' % i].T, sd['transformer.h.%d.attn.attention.k_proj.weight' % i].T, sd['transformer.h.%d.attn.attention.v_proj.weight' % i].T], axis=-1)
    # dumpmat(c_attn_w)

    dumpmat(sd['model.layers.%d.self_attn.q_proj.weight' % i].T)
    dumpmat(sd['model.layers.%d.self_attn.k_proj.weight' % i].T)
    dumpmat(sd['model.layers.%d.self_attn.v_proj.weight' % i].T)
            
    dumpmat(sd['model.layers.%d.self_attn.o_proj.weight' % i].T)

    dumpvec(sd['model.layers.%d.post_attention_layernorm.weight' % i])

    dumpmat(sd['model.layers.%d.mlp.gate_proj.weight' % i].T)
    dumpmat(sd['model.layers.%d.mlp.up_proj.weight' % i].T)
    dumpmat(sd['model.layers.%d.mlp.down_proj.weight' % i].T)
    
dumpvec(sd['model.norm.weight'])
dumpmat(sd['lm_head.weight'].T)


f.close()