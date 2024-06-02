import struct
from transformers import GPT2Model, GPT2LMHeadModel
from tqdm import tqdm

model = GPT2LMHeadModel.from_pretrained('gpt2')
sd = model.state_dict()

# print(sd['transformer.h.0.mlp.c_fc.weight'].shape)
# exit()

f = open('weights.dat', 'wb')

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

for i in tqdm(range(12)):
    dumpvec(sd['transformer.h.%d.ln_1.weight' % i])
    dumpvec(sd['transformer.h.%d.ln_1.bias' % i])
    dumpmat(sd['transformer.h.%d.attn.c_attn.weight' % i])
    dumpvec(sd['transformer.h.%d.attn.c_attn.bias' % i])
    dumpmat(sd['transformer.h.%d.attn.c_proj.weight' % i])
    dumpvec(sd['transformer.h.%d.attn.c_proj.bias' % i])
    dumpvec(sd['transformer.h.%d.ln_2.weight' % i])
    dumpvec(sd['transformer.h.%d.ln_2.bias' % i])
    dumpmat(sd['transformer.h.%d.mlp.c_fc.weight' % i])
    dumpvec(sd['transformer.h.%d.mlp.c_fc.bias' % i])
    dumpmat(sd['transformer.h.%d.mlp.c_proj.weight' % i])
    dumpvec(sd['transformer.h.%d.mlp.c_proj.bias' % i])
dumpvec(sd['transformer.ln_f.weight'])
dumpvec(sd['transformer.ln_f.bias'])
dumpmat(sd['lm_head.weight'].T)
f.close()