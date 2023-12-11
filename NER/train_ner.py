#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from NERDA.models import NERDA
import torch


def get_train_data(train_file):
    df = pd.read_pickle(train_file + ".pkl")
    training = {
        'sentences' : list(df['sentences']),
        'tags' : list(df['tags'])
    }

    validation = {
        'sentences' : list(df['sentences'])[0:1],
        'tags' : list(df['tags'])[0:1]
    }
    return training, validation


def get_tag_sheme_for_training():
    s = ['B-APPELLANT',  'I-APPELLANT',  'B-JUDGE',  'I-JUDGE',  'B-APPELLANT-COUNSEL',  'I-APPELLANT-COUNSEL',  'B-RESPONDENT-COUNSEL',  'I-RESPONDENT-COUNSEL',  'B-RESPONDENT',  'I-RESPONDENT',  'B-COURT',  'I-COURT',  'B-PRECEDENT',  'I-PRECEDENT',  'B-AUTHORITY',  'I-AUTHORITY',  'B-WITNESS',  'I-WITNESS']
    print("Tag Scheme",s)
    return s


def train_ner(train_file, model_name, epochs, transformer):
    training, validation = get_train_data(train_file)
    tag_scheme = get_tag_sheme_for_training()
#     transformer = 'nlpaueb/legal-bert-base-uncased'
#     transformer = 'dslim/bert-base-NER-uncased'
    print("model - ", transformer)
    dropout = 0.1
    training_hyperparameters = {
        'epochs' : 50,
        'warmup_steps' : 500,
        'train_batch_size': 16,
        'learning_rate': 0.0001,
    }
    model = NERDA(
        dataset_training = training,
        dataset_validation = validation,
        tag_scheme = tag_scheme, 
        tag_outside = 'O',
        transformer = transformer,
        dropout = dropout,
        hyperparameters = training_hyperparameters,
        max_len = 512,
        validation_batch_size = 8
        )
    
    model.train()
    model.save_network(model_name + '.bin')
    torch.save(model, model_name + '.pt')




