import sys
import os
sys.path.append(os.getcwd()+"/src/")
import pickle
from argparse import Namespace
from typing import Tuple, Dict
import random

import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F

from dataset import Seq2SeqDataset
from pytorch_lightning.callbacks import ModelCheckpoint

input_route = sys.argv[1]
output_route = sys.argv[2]

class Encoder(nn.Module):
    def __init__(self, embedding_path, emb_dim, enc_hid_dim, enc_layers, dec_hid_dim, enc_dropout):
        super(Encoder, self).__init__()
        with open(embedding_path, 'rb') as f:
            embedding = pickle.load(f)
        embedding_weight = embedding.vectors
        self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.enc_layers = enc_layers
        self.enc_dropout = enc_dropout
        
        self.rnn = nn.GRU(input_size = emb_dim,
                          hidden_size = enc_hid_dim,
                          num_layers = enc_layers,
                          bidirectional = True,
                          batch_first = False)
        
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(enc_dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        return outputs, hidden    

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)
    
class Decoder(nn.Module):
    def __init__(self, embedding_path, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dec_dropout, dec_num_layers, attention):
        super().__init__()

        with open(embedding_path, 'rb') as f:
            embedding = pickle.load(f)
        embedding_weight = embedding.vectors
        self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        self.output_dim = output_dim
        
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dec_dropout)
        self.attention = attention
        
    def forward(self, input, hidden, encoder_outputs):

        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))

        return prediction, hidden.squeeze(0)

class Seq2Seq_attn(pl.LightningModule):
    def __init__(self, hparams) -> None:
        super(Seq2Seq_attn, self).__init__()
        self.hparams = hparams
        self.criterion = nn.CrossEntropyLoss(ignore_index = self.hparams.ignore_idx)
        attn = Attention(hparams.enc_hid_dim,
                         hparams.dec_hid_dim)
        self.encoder = Encoder(hparams.embedding_path,
                               hparams.emb_dim,
                               hparams.enc_hid_dim,
                               hparams.enc_num_layers,
                               hparams.dec_hid_dim,
                               hparams.enc_dropout)
        self.decoder = Decoder(hparams.embedding_path,
                               hparams.output_dim,
                               hparams.emb_dim,
                               hparams.enc_hid_dim,
                               hparams.dec_hid_dim,
                               hparams.dec_dropout,
                               hparams.dec_num_layers,
                               attn)

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        batch_size = src.shape[1]
        trg_vocab_size = self.decoder.output_dim
                
        if trg != 'test':
            input = trg[0,:]
            trg_len = trg.shape[0]

            outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to('cuda:0')
            encoder_outputs, hidden = self.encoder(src)
            for t in range(1, trg_len):
                output, hidden = self.decoder(input, hidden, encoder_outputs)
                outputs[t] = output
                teacher_force = random.random() > teacher_forcing_ratio
                top1 = output.argmax(1) 
                input = trg[t] if teacher_force else top1
        else:
            input = torch.ones(batch_size).to(device = 'cuda:0', dtype=torch.int64)
            trg_len = 80
            outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to('cuda:0')
            encoder_outputs, hidden = self.encoder(src)
            for t in range(1, trg_len):
                output, hidden = self.decoder(input, hidden, encoder_outputs)
                outputs[t] = output
                input = output.argmax(1) 

        return outputs

    def _unpack_batch(self, batch) -> Tuple[torch.tensor, torch.tensor]:
        return batch['text']

    def _calculate_loss(self, output, trg) -> torch.tensor:
        # TODO
        # calculate the logits
        # plz use BCEWithLogit
        # adjust pos_weight!
        # MASK OUT PADDINGS' LOSSES!
        
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].reshape(-1)
        
        loss = self.criterion(output, trg)
        
        return loss

    def training_step(self, batch, batch_nb) -> Dict:
        x, y = self._unpack_batch(batch)
        x = x.permute(1,0)
        y = y.permute(1,0)
        output = self.forward(x,y)
        loss = self._calculate_loss(output, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb) -> Dict:
        x, y = self._unpack_batch(batch)
        x = x.permute(1,0)
        y = y.permute(1,0)
        output = self.forward(x,y)
        loss = self._calculate_loss(output, y)
        return {'val_loss': loss}
    
    def validation_epoch_end(self, outputs) -> Dict:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def _load_dataset(self, dataset_path: str) -> Seq2SeqDataset:
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
        
        
from tqdm import tqdm
PATH_checkpoint = "best_seq2seq_attn_model.ckpt"
seq2seq_attn = Seq2Seq_attn.load_from_checkpoint(PATH_checkpoint)

# if pl test doesnt work
with open(input_route, 'rb') as f:
    dataset = pickle.load(f)
test_loader = DataLoader(dataset, 
                         8, 
                         collate_fn=dataset.collate_fn)
device = 'cuda:0'
trange_test = tqdm(enumerate(test_loader), total=len(test_loader), desc = 'Test')
ans = []
seq2seq_attn.train(False)
log_softmax = nn.LogSoftmax(dim=-1)
for z, (batch) in trange_test:
    x = seq2seq_attn._unpack_batch(batch)
    x = x.to(device)
    x = x.permute(1,0)
    seq2seq_attn.to(device)
    output = seq2seq_attn.forward(x, 'test', 0)
    output = log_softmax(output)
    output = torch.argmax(output.permute(1,0,2), axis = 2)    
    output = output.type(torch.int64).tolist()
    ans.extend(output)
    
with open("./embedding_seq2seq.pkl", 'rb') as f:
    embed_dataset = pickle.load(f)
    
import numpy as np
ans_jsonl = []
for data, a in zip(dataset, ans):
    now_sent = ''
    for i in a:
        if i != 2:
            now_vocab = embed_dataset.vocab[i]
            now_sent = now_sent + now_vocab + ' '
        else:
            now_sent = now_sent[:-6]
            now_sent = now_sent[7:]
            break
    ans_jsonl.append({'id':data['id'], 'predict': now_sent})
    
import json
with open(output_route, 'w') as outfile:
    for entry in ans_jsonl:
        json.dump(entry, outfile)
        outfile.write('\n')