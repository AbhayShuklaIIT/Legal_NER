#!/usr/bin/env python
# coding: utf-8

import os
from utility import *
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
import torch
from tqdm.notebook import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
import numpy as np
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
from transformers import AdamW
from tqdm import tqdm

label_to_int_dict = get_label_to_int_dict()

def get_train_data(path):
    df = pd.read_pickle(path)
    x_train = df["x_train"]
    y_train = df["y_train"]
    y_train = list(y_train)
    df_train = pd.DataFrame.from_dict({"texts":x_train, "labels":y_train})
    df_train, df_val = np.split(df_train, [int(1*len(df_train))])
    print(len(df_train),len(df_val))
    return df_train, df_val, x_train, y_train



def get_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    new_tokens = ['<NE>', '</NE>']
    special_tokens_dict = {'additional_special_tokens': new_tokens}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = list(df['labels'])
        self.texts = list(df['texts'])

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


class BertClassifier(nn.Module):

    def __init__(self, device, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('nlpaueb/legal-bert-base-uncased')
        self.tokenizer = get_tokenizer()
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.linear = nn.Linear(768*1, 1)
        self.FL = nn.Sigmoid()
        self.device = device

    def forward(self, text):
        encodings = self.tokenizer(text,padding='max_length', max_length = 256, truncation=True,return_tensors="pt")
        encodings = encodings.to(self.device)
        result, _ = self.bert(input_ids= encodings["input_ids"], attention_mask=encodings["attention_mask"],return_dict=False)

        concat = result[:,0,:]
        linear_output = self.linear(concat)
        final_layer = self.FL(linear_output)
        return final_layer


def get_optimizer(model):
    pretrained = model.bert.parameters()
    pretrained_names = [f'bert.{k}' for (k, v) in model.bert.named_parameters()]
    new_params= [v for k, v in model.named_parameters() if k not in pretrained_names]
    optimizer = AdamW(
        [{'params': pretrained, 'lr' : 1e-5}, {'params': new_params, 'lr': 0.01}]
    )
    return optimizer

def get_sample_loss_weights(ip_batch, com):
    
    nc_wt_dict = {'APPELLANT' : 36.98706467661692,
            'JUDGE' : 50.78142076502732,
            'APPELLANT COUNSEL' : 66.85611510791367,
            'RESPONDENT COUNSEL' : 93.39698492462311,
            'RESPONDENT' : 35.949709864603484,
            'COURT' : 18.604604604604603,
            'PRECEDENT' : 6.770856102003643,
            'AUTHORITY' : 27.865067466266865,
            'WITNESS' : 57.012269938650306,
            'OTHER' : 1.550577733283277,}
    c_wt_dict = {'APPELLANT' : 57.37671232876712,
            'JUDGE' : 35.6468085106383,
            'APPELLANT COUNSEL' : 48.421965317919074,
            'RESPONDENT COUNSEL' : 51.70987654320987,
            'RESPONDENT' : 56.22147651006711,
            'COURT' : 32.343629343629345,
            'PRECEDENT' : 3.853265869365225,
            'AUTHORITY' : 41.47029702970297,
            'WITNESS' : 94.12359550561797,
            'OTHER' : 1.7495822890559731,}
    if com == 0:
        wt_dict = nc_wt_dict
    else:
        wt_dict = c_wt_dict
        
    wts = []
    
    for i in ip_batch:
        wts.append(wt_dict[i[:i.find("[SEP]")]])
    return wts

def train_model(data_path,save_model_name, epochs, combined, use_cuda = 0, batch_size = 16, model_load = ''):

    device = torch.device("cuda" if use_cuda else "cpu")
    
    model = BertClassifier(device).to(device)
    if model_load != '':model = torch.load( model_load, map_location=torch.device(device))

    train_data, val_data, x_train, y_train = get_train_data(data_path)
    optimizer = get_optimizer(model)
    
    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)

    criterion = nn.BCELoss(reduction='none')

    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):
                train_label = train_label.to(device)
                sample_wts = get_sample_loss_weights(train_input, combined)
                sample_wts = torch.Tensor(sample_wts).to(device)
                output = model(train_input)


                train_label = train_label.to(torch.float)
                # print(output.shape, train_label.unsqueeze(1).shape)
                intermediate_losses  = criterion(output, train_label.unsqueeze(1))
            
#                 print("wts and inter loss",sample_wts,intermediate_losses.squeeze())
                batch_loss = torch.mean(sample_wts*intermediate_losses)
#                 print("wt loss", batch_loss)
                total_loss_train += batch_loss.item()


                # print(((output>0.5) == train_label.unsqueeze(1)))
                # print((output>0.5))
                # print(train_label.unsqueeze(1))
                acc = ((output>0.5) == train_label.unsqueeze(1)).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in tqdm(val_dataloader):

                    val_label = val_label.to(device)

                    output = model(val_input)
                    val_label = val_label.to(torch.float)

                    batch_loss = criterion(output, val_label.unsqueeze(1))
                    total_loss_val += batch_loss.item()
                    
                    
                    acc = ((output>0.5) == val_label.unsqueeze(1)).sum().item()
                    total_acc_val += acc
            
            print("saving model - " + save_model_name)
            torch.save(model, save_model_name) 
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f}')# \
