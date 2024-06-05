import torch
from torch import nn
import numpy as np
from torch.optim import Adam
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from datasets import load_dataset

tokenizer = BertTokenizer.from_pretrained('yoshitomo-matsubara/bert-large-uncased-mrpc')
model = AutoModelForSequenceClassification.from_pretrained("yoshitomo-matsubara/bert-large-uncased-mrpc")

sst2_val = load_dataset('SetFit/mrpc', split='validation')

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = [d['label'] for d in df]
        self.texts = [
            tokenizer(d['text1'],d['text2'],
                padding='max_length', 
                max_length = 512, 
                truncation=True,
                return_tensors="pt")
            for d in df
        ]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

model.eval()

total_acc_val = 0
val_dataloader = torch.utils.data.DataLoader(Dataset(sst2_val), batch_size=8)
# val_dataloader = torch.utils.data.DataLoader(Dataset(sst2_val), batch_size=32)
# for x in val_dataloader:
#     print(x)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = model.to(device)

tot = 0
with torch.no_grad():

    for val_input, val_label in val_dataloader:
        tot += len(val_label)
        val_label = val_label.to(device)
        mask = val_input['attention_mask'].to(device)
        # token_type_ids = val_input['token_type_ids'].squeeze(1).to(device)
        input_id = val_input['input_ids'].squeeze(1).to(device)

        # output = model(input_id, mask, token_type_ids)
        output = model(input_id, mask)
        
        acc = (output.argmax(dim=1) == val_label).sum().item()
        total_acc_val += acc        

print(total_acc_val, "/", tot, "=", 100*total_acc_val/tot, "%")