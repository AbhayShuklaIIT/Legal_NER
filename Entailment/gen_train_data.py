from utility import *
import pandas as pd
import glob
import random
import os
import shutil


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

def negative_sample(ne, context, roles, label_to_int_dict):
    no_ns_per = 4
    x = []
    y = []
    
    if len("".join(context))<10:return [],[] 
    
    for role in roles:
        if role not in label_to_int_dict.keys():continue
        for c in context:
            x.append(role + "[SEP]" + c)
            y.append(1)
            neg_labels = []
            while len(neg_labels) != no_ns_per:
                random_label = random.choice(list(label_to_int_dict.keys()))
                if random_label in roles or random_label in neg_labels:continue
                neg_labels.append(random_label)
                x.append(random_label + "[SEP]" + c)
                y.append(0)
    return x,y


        
def get_x_y_per_doc(NEs, contexts, roles, label_to_int_dict):
    x_f = []
    y_f = []
    for ne, context, role in zip(NEs, contexts, roles):
        x_i, y_i = negative_sample(ne, context, role, label_to_int_dict)
        x_f.extend(x_i)
        y_f.extend(y_i)
    return x_f, y_f

def get_x_y(label_to_int_dict, combine_ctx, test_req = 0):
    NEs, contexts, roles, spacy_labels, NEs_test, contexts_test, roles_test, spacy_labels_test = get_data(combine_ctx = combine_ctx, test_req = test_req)
    roles = correct_roles(roles)
    roles_test = correct_roles(roles_test)
    x, y_f = get_x_y_per_doc(NEs, contexts, roles, label_to_int_dict)
    if test_req == 1:
        x_test, y_f_test = get_x_y_per_doc(NEs_test, contexts_test, roles_test, label_to_int_dict)
    else:
        x_test, y_f_test = [], []
    
    return (x), (y_f), (x_test), (y_f_test), label_to_int_dict

def gen_training_data_entailment(train_file ,combine_ctx, test_req = 0):
    print("Combined", combine_ctx)
    label_to_int_dict = get_label_to_int_dict()
    print("Label to int dict", label_to_int_dict)
    x_train, y_train, x_test, y_test, label_to_int_dict = get_x_y(label_to_int_dict, test_req = test_req, combine_ctx = combine_ctx)
    df_dict = {"x_train" : x_train, "y_train" : y_train}
    df = pd.DataFrame.from_dict(df_dict)
    print("Saving data to " + train_file + ".pkl")
    df.to_pickle(train_file + ".pkl")