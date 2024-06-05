import numpy as np
from datasets import load_dataset
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import struct
import sys

tokenizer = BertTokenizer.from_pretrained('bert-large-cased')

def tokenize(tokenizer, text_a, text_b=None):
    tokens_a = ["[CLS]"] + tokenizer.tokenize(text_a) + ["[SEP]"]
    tokens_b = (tokenizer.tokenize(text_b) + ["[SEP]"]) if text_b else []

    tokens = tokens_a + tokens_b
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(tokens_a) + [1] * len(tokens_b)

    return tokens, input_ids, segment_ids

f = None

def set_file(filename):
    global f
    f = open(filename, 'wb')

def close_file():
    global f
    f.close()
    f = None

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

def main():

    # load dataset
    dataset = load_dataset("sst2", split="validation")
    model = AutoModelForSequenceClassification.from_pretrained("yoshitomo-matsubara/bert-large-uncased-mrpc")
    sd = model.state_dict()

    wse = sd["bert.embeddings.token_type_embeddings.weight"]
    wte = sd["bert.embeddings.word_embeddings.weight"]
    wpe = sd["bert.embeddings.position_embeddings.weight"]

    max_len = 512
    n_head = 2

    g = open("labels.txt","w")

    i = 0
    for text, label in tqdm(zip(dataset["sentence"], dataset["label"])):
        _, input_ids, segment_ids = tokenize(tokenizer, text)
        input_ids, segment_ids = input_ids[:max_len], segment_ids[:max_len]
        x = wte[input_ids] + wpe[range(len(input_ids))] + wse[segment_ids]
        # print(x.shape)
        g.write(str(label) + "\n")
        set_file('./dataset/' + str(i) + '.dat')
        dumpmat(x)
        close_file()
        # print(label)
        i += 1
    
    g.close()



if __name__ == "__main__":
    main()