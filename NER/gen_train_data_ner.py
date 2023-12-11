#!/usr/bin/env python
# coding: utf-8

from utility_ner import *
import glob as glob
import nltk
import pandas as pd
import tqdm as tqdm
import shutil
import os


def split_train_test(train, test):
    print("Making train and test")
    shutil.rmtree("./train/")
    os.mkdir("./train/")
    for i in train:
        name = i[i.rfind("/")+1:]
#         print(name)
        df = pd.read_excel(i,index_col = 0, na_filter = False)
        df.to_excel("./train/"+name)
    shutil.rmtree("./test/")
    os.mkdir("./test/")
    for i in test:
        name = i[i.rfind("/")+1:]
#         print(name)
        df = pd.read_excel(i,index_col = 0, na_filter = False)
        df.to_excel("./test/"+name)


def nest_sentences(document,chunk_length):
    nested = []
    sent = []
    length = 0
    for sentence in nltk.sent_tokenize(document):
        length += len(sentence.split(" "))
        if length < chunk_length:
            sent.append(sentence)
        else:
            nested.append(' '.join(sent))
            sent = []
            sent.append(sentence)
            length = len(sentence.split(" "))
    if len(sent)>0:
        nested.append(' '.join(sent))
    return nested

def return_sents_file(path, label_to_int_dict):
    f = open(path)
    data = json.load(f)
    sents = []
    for i in data:
        sen = ''
        for j in (data[i]):
            sents.append(' '.join(data[i][j]))
    # print(data)
    r = nest_sentences(' '.join(sents), 128)
    return r


def get_pos_in_sen(ne, sent, label_to_int_dict):
    ne_words = nltk.word_tokenize(ne)
    pos_s_l = []
    pos_e_l = []
    pos_s = -1
    pos_e = -1
    for i in range(len(sent)):
        i_word = sent[i]
        if i_word.lower() == ne_words[0].lower():
            flag = 0
            for j in range(len(ne_words)):
                if i+j >= len(sent) or sent[i+j].lower() != ne_words[j].lower():
                    flag = 1
                    break
            if flag == 0:
                pos_s = i
                pos_e = i + len(ne_words)
                pos_s_l.append(pos_s)
                pos_e_l.append(pos_e)
    return pos_s_l, pos_e_l


def get_labelv2(label, label_to_int_dict, val = 0):
    ans = get_label(label)
    ans = ans.replace(" ", "-")
    if ans == "OTHER": ans = ""
    if ans not in list(label_to_int_dict.keys()):
        if ans != "":print("Error in label", ans)
        ans = ""
    return ans


from collections import defaultdict
def get_ne_groups_and_labels(df, path, label_to_int_dict):
    NEs = []
    roles = []
    spacy_labels = []
    df.columns = [i.lower() for i in df.columns]
    cols = ['entities', 'labels', 'frequency', 'variant', 'role']
    df = df[cols]
    df = remove_nan(df)
    r_l = list(df["role"])
    s_l = list(df["labels"])
    nes_l = list(df["entities"])
    varient_l = list(df["variant"])
    f_l = list(df["frequency"])
    
    ne_dict = defaultdict(list)
    map_ne_label = {}
    map_ne_an_label = {}

    for i in range(len(nes_l)):
        try:
            ne = nes_l[i]
            r_t = r_l[i]
            p_i = int(i)
            patience = 10
            while len(r_t) == 0:
                if patience == -1: break
                try:
                    p_i = int(varient_l[p_i])
                except:
                    p_i = varient_l[p_i].split(",")
                    try:
                        p_i = int(p_i[0])
                    except Exception as e:
                        print(e)
                        p_i = i
                        break

                r_t = r_l[p_i]
                patience = patience  - 1
            if len(r_t) == 0:continue
            map_ne_an_label[ne] = get_labelv2(r_t, label_to_int_dict)
        except Exception as e:
            print("s1 ", e)
    return ne_dict, map_ne_label, map_ne_an_label, list(nes_l)

def process_sent(sent, nes, map_ne_an_label, label_to_int_dict):
    sents_words = nltk.word_tokenize(sent)
    tags = ['O' for i in sents_words]
    for ne in nes:
        try:
            if map_ne_an_label[ne] in skip:continue
        except Exception as e:
            # print("s.1", e, ne)
            continue
        pos_s_l, pos_e_l = get_pos_in_sen(ne, sents_words, label_to_int_dict)
        for it in range(len(pos_s_l)):
            pos_s = pos_s_l[it]
            pos_e = pos_e_l[it]
            if pos_s == -1:continue
            tags[pos_s] = "B-" + map_ne_an_label[ne]
            c = pos_s + 1
            while c < pos_e:
                tags[c] = "I-" + map_ne_an_label[ne]
                c = c + 1
    flag = 1
    for i in tags:
        if i != "O":flag = 0
    if flag == 1:return [],[]
    return sents_words, tags


def get_sents_tags(file, label_to_int_dict):
    try:
        ne_dict, map_ne_label, map_ne_an_label, nes = get_ne_groups_and_labels(pd.read_excel(file, index_col = 0, na_filter = None), file, label_to_int_dict)
    except Exception as e:
        print("s10 ", e)
        return [], []
    sents = return_sents_file("./jsons/" + file[file.rfind("/")+1:-5] + ".json", label_to_int_dict)
    sents_per_file = []
    tags_per_file = []
    for sen in sents:
        s, t = process_sent(sen, nes, map_ne_an_label, label_to_int_dict)
        if len(s) == 0:continue
        sents_per_file.append(s)
        tags_per_file.append(t)
    return sents_per_file, tags_per_file


def get_ner_data(files, label_to_int_dict):
    sents = []
    tags = []
    for file in tqdm.tqdm(files):
        print(file)
        try:
            s,t = get_sents_tags(file, label_to_int_dict)
            sents.extend(s)
            tags.extend(t)
        except Exception as e:
            print("s10", e)
    return {
        'sentences' : sents,
        'tags' : tags
    }

def get_ner_data_and_save(train_file):
    label_to_int_dict = get_label_to_int_dict()
    train_files = glob.glob("./train/*.xlsx")
    train_data = get_ner_data(train_files, label_to_int_dict)
    df = pd.DataFrame.from_dict(train_data)
    print("saving train data in", train_file + ".pkl")
    df.to_pickle(train_file + ".pkl")




