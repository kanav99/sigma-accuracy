import torch
import struct
import numpy as np
from tqdm import tqdm
import sys
from transformers import GPTNeoForCausalLM

model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
# model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
sd = model.state_dict()

for k in sd.keys():
    sd[k] = sd[k].cpu()

# f = open('bert_sst2_90_t.dat', 'wb')
f = open("weights.dat", 'wb')

def dumpvec(x):
    # print(x.shape)
    assert(len(x.shape) == 1)
    CI = x.shape[0]
    for ci in range(CI):
        f.write(struct.pack('f', x[ci].item()))

def dumpmat(w):
    # print(w.shape)
    assert(len(w.shape) == 2)
    CI = w.shape[0]
    CO = w.shape[1]
    for ci in range(CI):
        for co in range(CO):
            f.write(struct.pack('f', w[ci][co].item()))

for i in tqdm(range(32)):

    dumpvec(sd['transformer.h.%d.ln_1.weight' % i])
    dumpvec(sd['transformer.h.%d.ln_1.bias' % i])

    # c_attn_w = np.concatenate([sd['transformer.h.%d.attn.attention.q_proj.weight' % i].T, sd['transformer.h.%d.attn.attention.k_proj.weight' % i].T, sd['transformer.h.%d.attn.attention.v_proj.weight' % i].T], axis=-1)
    # dumpmat(c_attn_w)

    dumpmat(sd['transformer.h.%d.attn.attention.k_proj.weight' % i].T)
    dumpmat(sd['transformer.h.%d.attn.attention.v_proj.weight' % i].T)
    dumpmat(sd['transformer.h.%d.attn.attention.q_proj.weight' % i].T)
            
    dumpmat(sd['transformer.h.%d.attn.attention.out_proj.weight' % i].T)
    dumpvec(sd['transformer.h.%d.attn.attention.out_proj.bias' % i])

    dumpvec(sd['transformer.h.%d.ln_2.weight' % i])
    dumpvec(sd['transformer.h.%d.ln_2.bias' % i])

    dumpmat(sd['transformer.h.%d.mlp.c_fc.weight' % i].T)
    dumpvec(sd['transformer.h.%d.mlp.c_fc.bias' % i])
    dumpmat(sd['transformer.h.%d.mlp.c_proj.weight' % i].T)
    dumpvec(sd['transformer.h.%d.mlp.c_proj.bias' % i])
    
dumpvec(sd['transformer.ln_f.weight'])
dumpvec(sd['transformer.ln_f.bias'])
dumpmat(sd['lm_head.weight'].T)


f.close()