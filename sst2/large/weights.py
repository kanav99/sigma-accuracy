import torch
import struct
import numpy as np
from tqdm import tqdm
import sys
from transformers import AutoModelForSequenceClassification

# sd = torch.load(sys.argv[1], map_location=torch.device('cpu'))
model = AutoModelForSequenceClassification.from_pretrained("yoshitomo-matsubara/bert-large-uncased-sst2")
sd = model.state_dict()

for k in sd.keys():
    sd[k] = sd[k].cpu()

# f = open('bert_sst2_90_t.dat', 'wb')
# f = open(sys.argv[2], 'wb')
f = open("weights.dat", 'wb')

def dumpvec(x):
    # print(x.shape)
    assert(len(x.shape) == 1)
    CI = x.shape[0]
    try:
        x = x.numpy()
    except:
        pass
    f.write(x.tobytes())

def dumpmat(w):
    # print(w.shape)
    assert(len(w.shape) == 2)
    CI = w.shape[0]
    CO = w.shape[1]
    try:
        w = w.numpy()
    except:
        pass
    f.write(w.tobytes())

dumpvec(sd["bert.embeddings.LayerNorm.weight"])
dumpvec(sd["bert.embeddings.LayerNorm.bias"])

for i in tqdm(range(24)):
    pref = "bert.encoder.layer." + str(i)
    c_attn_w = np.concatenate([sd[pref + ".attention.self.query.weight"].T, sd[pref + ".attention.self.key.weight"].T, sd[pref + ".attention.self.value.weight"].T], axis=-1)
    c_attn_b = np.concatenate([sd[pref + ".attention.self.query.bias"], sd[pref + ".attention.self.key.bias"], sd[pref + ".attention.self.value.bias"]], axis=-1)
    # print(c_attn_w.shape)
    dumpmat(c_attn_w)
    dumpvec(c_attn_b)
    c_proj_w = sd[pref + ".attention.output.dense.weight"].T
    c_proj_b = sd[pref + ".attention.output.dense.bias"]
    # print(c_proj_w.shape)
    dumpmat(c_proj_w)
    dumpvec(c_proj_b)
    ln_0_w = sd[pref + ".attention.output.LayerNorm.weight"]
    ln_0_b = sd[pref + ".attention.output.LayerNorm.bias"]
    dumpvec(ln_0_w)
    dumpvec(ln_0_b)
    mlp_w = sd[pref + ".intermediate.dense.weight"].T
    mlp_b = sd[pref + ".intermediate.dense.bias"]
    dumpmat(mlp_w)
    dumpvec(mlp_b)
    # print(mlp_w.shape)
    
    mlp_w = sd[pref + ".output.dense.weight"].T
    mlp_b = sd[pref + ".output.dense.bias"]
    # print(mlp_w.shape)
    dumpmat(mlp_w)
    dumpvec(mlp_b)
    ln_1_w = sd[pref + ".output.LayerNorm.weight"]
    ln_1_b = sd[pref + ".output.LayerNorm.bias"]
    dumpvec(ln_1_w)
    dumpvec(ln_1_b)
    
dumpmat(sd["bert.pooler.dense.weight"].T)
dumpvec(sd["bert.pooler.dense.bias"])
# dumpmat(sd["linear.weight"].T)
# dumpvec(sd["linear.bias"])
dumpmat(sd["classifier.weight"].T)
dumpvec(sd["classifier.bias"])

f.close()