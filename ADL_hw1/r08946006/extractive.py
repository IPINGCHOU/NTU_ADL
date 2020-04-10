import sys
sys.path.append("./src")

import os
import pickle
from argparse import Namespace
from typing import Tuple, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F

from dataset import SeqTaggingDataset
from pytorch_lightning.callbacks import ModelCheckpoint

input_route = sys.argv[1]
output_route = sys.argv[2]

#%%
class Encoder(nn.Module):
    def __init__(self,
                 embedding_path,
                 embed_size,
                 rnn_hidden_size,
                 rnn_num_layers) -> None:
        super(Encoder, self).__init__()
        with open(embedding_path, 'rb') as f:
            embedding = pickle.load(f)
        embedding_weight = embedding.vectors
        self.embedding = nn.Embedding.from_pretrained(embedding_weight)

        # TODO (init a LSTM/RNN)
        self.rnn = nn.GRU(input_size = embed_size,
                          hidden_size = rnn_hidden_size,
                          num_layers = rnn_num_layers,
                          bidirectional=True,
                          batch_first=True)

    def forward(self, idxs) -> Tuple[torch.tensor, torch.tensor]:
        embed = self.embedding(idxs)
        output, state = self.rnn(embed)
        return output, state

class SeqTagger(pl.LightningModule):
    def __init__(self, hparams) -> None:
        super(SeqTagger, self).__init__()
        self.hparams = hparams
        self.criterion = nn.BCEWithLogitsLoss(
            reduction='mean', 
            pos_weight=torch.tensor(hparams.pos_weight))
        self.encoder = Encoder(hparams.embedding_path,
                               hparams.embed_size,
                               hparams.rnn_hidden_size,
                               hparams.rnn_num_layers)
        self.proj_1 = nn.Linear(2*self.encoder.rnn.hidden_size, 256)
        self.proj_2 = nn.Linear(256, 128)
        self.proj_3 = nn.Linear(128,1)
        
        self.ans = []

    def forward(self, idxs) -> torch.tensor:
        # TODO
        # take the output of encoder
        # project it to 1 dimensional tensor
        enc_output, state = self.encoder(idxs)
        l1 = F.relu(self.proj_1(enc_output))
        l2 = F.relu(self.proj_2(l1))
        
        logits = self.proj_3(l2)
    
        return logits

    def _unpack_batch(self, batch) -> Tuple[torch.tensor, torch.tensor]:
        return batch['text']

    def _calculate_loss(self, y_hat, y) -> torch.tensor:
        # TODO
        # calculate the logits
        # plz use BCEWithLogit
        # adjust pos_weight!
        # MASK OUT PADDINGS' LOSSES!
        y_hat = y_hat.reshape(-1)
        y = y.reshape(-1)
        non_ignore = torch.where(y!= self.hparams.ignore_idx)[0]    

        loss = self.criterion(y_hat[non_ignore], y[non_ignore])
        
        return loss

    def training_step(self, batch, batch_nb) -> Dict:
        x, y = self._unpack_batch(batch)
        logit = self.forward(x)
        loss = self._calculate_loss(logit, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb) -> Dict:
        x, y = self._unpack_batch(batch)
        logit = self.forward(x)
        loss = self._calculate_loss(logit, y)
        return {'val_loss': loss}
    
    def test_step(self, batch, batch_nb) -> list:
        x = self._unpack_batch(batch)
        logit = self.forward(x)
        pred = torch.sigmoid(logit)
        pred = pred > self.hparams.threshold
        b,s,w = pred.shape
        pred = pred.view(b,s)
        pred = pred.type(torch.uint8).tolist()
        return pred
        
    def test_epoch_end(self, outputs) -> Dict:
        for pred in outputs:
            self.ans.extend(pred)
        print('Now threshold: ' + str(self.hparams.threshold))
        return {'test': 0}
    
    def validation_epoch_end(self, outputs) -> Dict:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def _load_dataset(self, dataset_path: str) -> SeqTaggingDataset:
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset

    def train_dataloader(self):
        dataset = self._load_dataset(self.hparams.train_dataset_path)
        return DataLoader(dataset, 
                          self.hparams.batch_size, 
                          shuffle=True,
                          collate_fn=dataset.collate_fn)

    def val_dataloader(self):
        dataset = self._load_dataset(self.hparams.valid_dataset_path)
        return DataLoader(dataset, 
                          self.hparams.batch_size, 
                          collate_fn=dataset.collate_fn)
    
    def test_dataloader(self):
        dataset = self._load_dataset(self.hparams.test_dataset_path)
        return DataLoader(dataset, 
                          self.hparams.test_batch_size, 
                          collate_fn=dataset.collate_fn)
    
#%%


#%%
from tqdm import tqdm
PATH_checkpoint = "./"
PATH_checkpoint += "best_extractive_model.ckpt"
seq_tagger = SeqTagger.load_from_checkpoint(PATH_checkpoint)

with open(input_route, 'rb') as f:
    dataset = pickle.load(f)
test_loader = DataLoader(dataset, 
                         32, 
                         collate_fn=dataset.collate_fn)

device = 'cuda:0'
trange_test = tqdm(enumerate(test_loader), total=len(test_loader), desc = 'Test')
ans = []
for z, (batch) in trange_test:
    x = seq_tagger._unpack_batch(batch)
    x = x.to(device)
    seq_tagger.to(device)
    logit = seq_tagger.forward(x)
    pred = torch.sigmoid(logit)
    pred = pred > 0.5 #THRESHOLD
    b,s,w = pred.shape
    pred = pred.view(b,s)
    pred = pred.type(torch.uint8).tolist()
    ans.extend(pred)
    
# Ratio of "1" should bigger than THRESHOLD_chosed%
import numpy as np
ans_jsonl = []
THRESHOLD_chosed = 0.1
for data, a in zip(dataset, ans):
    ratio_ones = []
    for i in data['sent_range']:
        now_sent = a[i[0]: i[1]]
        if len(now_sent)!=0:
            ratio_ones.append(np.sum(now_sent) / len(now_sent))
        else:
            continue
        
    pos = np.where(np.array(ratio_ones) >= THRESHOLD_chosed)[0].tolist()
    ans_jsonl.append({'id':data['id'],'predict_sentence_index':pos})
    
import json
ans_jsObj = json.dumps(ans_jsonl) 

with open(output_route, 'w') as outfile:
    for entry in ans_jsonl:
        json.dump(entry, outfile)
        outfile.write('\n')