import numpy as np
from datasets import load_dataset
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import struct
import sys

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf')

f = None

def set_file(filename):
    global f
    f = open(filename, 'wb')

def close_file():
    global f
    f.close()
    f = None

def dumpvec(x):
    global f
    if f == None:
        return
    assert(len(x.shape) == 1)
    CI = x.shape[0]
    for ci in range(CI):
        f.write(struct.pack('f', x[ci].item()))

def dumpmat(w):
    global f
    if f == None:
        return
    assert(len(w.shape) == 2)
    CI = w.shape[0]
    CO = w.shape[1]
    # for ci in range(CI):
    #     for co in range(CO):
    #         f.write(struct.pack('f', w[ci][co].item()))
    f.write(w.numpy().tobytes())

def main():

    # load dataset
    dataset = load_dataset('EleutherAI/lambada_openai', 'en', split='test')
    sd = model.state_dict()

    wte = sd["model.embed_tokens.weight"]

    max_len = 2048
    n_head = 12

    g = open("labels.txt","w")

    i = 0
    for d in tqdm(dataset):
        prompt = d['text']
        prompt_modified = prompt.rsplit(' ', 1)[0]

        original_last_word = prompt.rsplit(' ', 1)[1]
        original_last_word_token = tokenizer(" " + original_last_word, return_tensors="pt")['input_ids'][0][0].item()
        g.write(str(original_last_word_token) + "\n")
        
        inputs = tokenizer(prompt_modified, return_tensors="pt")
        input_ids = [x.item() for x in inputs['input_ids'][0]]
        input_ids = input_ids[:max_len]
        x = wte[input_ids]
        # print(x.shape)
        # set_file('../datasets/lambada-gpt2/' + str(i) + '.dat')
        set_file('dataset/' + str(i) + '.dat')
        dumpmat(x)
        close_file()
        # print(label)
        i += 1
    
    g.close()



if __name__ == "__main__":
    main()