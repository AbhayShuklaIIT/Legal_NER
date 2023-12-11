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
import re, math
from collections import Counter
from collections import defaultdict
import json
import pickle
seed_everything(42)

model_cache = {}

def get_list_from_labels_v1(roles, label_to_int_dict):
    y = []
    keys = list(label_to_int_dict.keys())
    for i in range(len(keys)):
        if keys[i] in roles:
            y.append(1)
        else:
            y.append(0)
    return y
def get_labels_from_ctx_v1(ne, contexts, label_to_int_dict, model, combined, use_cache):
    global model_cache
    int_to_label_dict = dict([(value, key) for key, value in label_to_int_dict.items()])
    probs = []
    model_labels = []
    if len(contexts) == 0:return []
    ip = []
    for ctx in contexts:
        ip.append(ne + "[SEP]" + ctx)
    if combined == 1:
        ip = []
        ip.append(ne + "[SEP]" + ". ".join(contexts))
    with torch.no_grad():
        if use_cache == 1 and tuple(ip) in model_cache.keys():
            p = model_cache[tuple(ip)]
        else:
            p = model(ip)
            if tuple(ip) not in model_cache.keys():model_cache[tuple(ip)] = p
        if use_cache == 2:
            model_cache[tuple(ip)] = p
#     print(np.array(p))
    a = torch.mean(p, 0)
    i_max = (torch.argmax(a))
    model_probs = [0 for _ in range(len(a))]
    model_probs[i_max]  = 1
#     print(model_probs)
    return model_probs

def convert_ner_dict_to_parent_to_child_dict(variants_dict):
    parent_to_child_dict = defaultdict(list)
    for i in variants_dict.keys():
        parent_to_child_dict[variants_dict[i]].append(i)
    return parent_to_child_dict

def get_parent_to_child_using_similarity(n):
    thres = 0.85
    ne_varient_dict = defaultdict(list)
    done = []
    for n_i in n:
        if n_i in done:continue
        ne_varient_dict[n_i].append(n_i)
        for n_j in n:
            if n_i == n_j:continue
            if get_similarity_1(n_i,n_j) > thres and get_similarity_2(n_i, n_j) > thres*100:
                ne_varient_dict[n_i].append(n_j)
                done.append(n_j)
#                 print(n_i, n_j)
        done.append(n_i)
    return ne_varient_dict

def get_ne_role_dict(nes, roles):
    ne_role_dict = {}
    for i in range(len(nes)):
        ne_role_dict[nes[i]] = roles[i]
    return ne_role_dict

def get_ne_to_freq(nes, fl):
    ne_to_freq = {}
    for i in range(len(nes)):
        ne_to_freq[nes[i]] = fl[i]
    return ne_to_freq

def evaluate_post_v1(path, label_to_int_dict, model, combined, use_cache, removed=[]):
    nes_r, ctx_r, roles_r, sl_r, fl_r, variants_dict = generate_training_examples(pd.read_excel(path, index_col = 0, na_filter = None), path)
    
    roles_r_c = correct_roles(roles_r)
    ne_to_freq = get_ne_to_freq(nes_r, fl_r)
    ne_role = get_ne_role_dict(nes_r, roles_r_c)
    parent_to_child_using_similarity = get_parent_to_child_using_similarity(nes_r)
    
    nes, ctx, roles, sl, fl = combine_variants(nes_r, ctx_r, roles_r, sl_r, fl_r, combined, variants_dict, 0)
    roles = correct_roles(roles)
    text = return_text_file("./jsons/"+path[path.rfind("/")+1:-5]+".json")
    yhat = []
    ytest = []
    for i in range(len(nes)):
        ne = nes[i]
        cs = ctx[i]
        rs = roles[i]
        
        flag = 0
        for r in removed:
            if r in rs:
                flag = 1
        if flag == 1: continue
        if len(rs) == 0:continue
        if len(cs) == 0:continue
        
        ytest_added = 0
        for child in parent_to_child_using_similarity[ne]:
            rs_ints = get_list_from_labels_v1(ne_role[child], label_to_int_dict)
            freq_child = ne_to_freq[child]
            if freq_child == "":
                freq_child = text.count(child)
            ytest.extend(freq_child * [rs_ints])
            ytest_added =ytest_added + freq_child 
        
        if fl[i] != "":
            pred_ints = get_labels_from_ctx_v1(ne, cs, label_to_int_dict, model, combined, use_cache)
            freq = fl[i]
        else:
            pred_ints = [0 for i in range(len(label_to_int_dict.keys()))]
            freq = text.count(ne)
        
        freq = ytest_added
        yhat.extend([pred_ints]*freq)
        if sum(rs_ints) == 0:
            print(rs)
            continue
    
    return ytest, yhat

def evaluate_v1(label_to_int_dict, model, combined, use_cache):
    train = glob.glob("./train/*.xlsx")
    test = glob.glob("./test/*.xlsx")
    yhat = []
    ytest = []
    for i in tqdm(test):
#         try:
        print(i)
        yt, yh = evaluate_post_v1(i, label_to_int_dict, model, combined, use_cache)
        if len(yt) != len(yh):
            print(i, "not sam len")
        ytest.extend(yt)
        yhat.extend(yh)
    return ytest, yhat


def del_labels(ytest, yhat, label_to_int_dict):
    label_to_int_dict = get_label_to_int_dict()
    ytest_del = np.delete(ytest, 9, 1)
    yhat_del = np.delete(yhat, 9, 1)
    labels = list(label_to_int_dict.keys())
    labels.remove('OTHER')
    return ytest_del, yhat_del, labels

def print_cls_report_and_vals(ytest_f, yhat_f, labels, thres = 0.5):
    col = len(labels) +4
    yhat_ct = np.array(yhat_f) > thres
    print(classification_report(np.array(ytest_f), yhat_ct, target_names = labels, digits=4))
    f1 = (classification_report(np.array(ytest_f), yhat_ct, target_names = labels).split("\n")[col].split("      ")[4])
    p = (classification_report(np.array(ytest_f), yhat_ct, target_names = labels).split("\n")[col].split("      ")[2])
    r = (classification_report(np.array(ytest_f), yhat_ct, target_names = labels).split("\n")[col].split("      ")[3])
    return p, r, f1


def print_result_v1(model_name, combined, device, use_cache, cache_file):
    global model_cache
    if use_cache == 2:model_cache = {}
    
    if use_cache == 1:
        with open(cache_file + '.pkl', 'rb') as f:
            model_cache = pickle.load(f)
    
    label_to_int_dict = get_label_to_int_dict()
    model = torch.load( model_name, map_location=torch.device(device)) 
    ytest, yhat = evaluate_v1(label_to_int_dict, model, combined, use_cache)
    ytest_n, yhat_n, labels = del_labels(ytest, yhat, label_to_int_dict)
    p, r, f = print_cls_report_and_vals(ytest_n, yhat_n, labels)
    
    if use_cache == 2 or use_cache == 1:
        with open(cache_file + '.pkl', 'wb') as f:
            pickle.dump(model_cache, f)
    return p, r, f


def jaccard_similarity(list1, list2):
#     print()
#     print(list1, list2, sep = "--\n--")
    list1 = [i.upper() for i in list1]
    list2 = [i.upper() for i in list2]
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    if union == 0: return 0 #union 0
    return float(intersection) / union


def evaluate_doc_v2(path, label_to_int_dict, model, combined,use_cache, removed = []):
    print(path)
    
    int_to_label_dict = dict([(value, key) for key, value in label_to_int_dict.items()])
    nes_r, ctx_r, roles_r, sl_r, fl_r, variants_dict = generate_training_examples(pd.read_excel(path, index_col = 0, na_filter = None), path)
    
    roles_r_c = correct_roles(roles_r)
    
    ne_role_dict_raw = get_ne_role_dict(nes_r, roles_r_c)
    parent_to_child_dict_sim = get_parent_to_child_using_similarity(nes_r)
    
    
    nes, ctx, roles, sl, fl = combine_variants(nes_r, ctx_r, roles_r, sl_r, fl_r, combined, variants_dict, 0)
    nes_g, ctx_g, roles_g, sl_g, fl_g = combine_variants(nes_r, ctx_r, roles_r, sl_r, fl_r, combined, variants_dict, 1)

    parent_to_child_dict_gold = convert_ner_dict_to_parent_to_child_dict(variants_dict)
    
#     print(parent_to_child_dict_sim)
#     print(parent_to_child_dict_gold)
    
    roles = correct_roles(roles)
    roles_g = correct_roles(roles_g)
    
    text = return_text_file("./jsons/"+path[path.rfind("/")+1:-5]+".json")
    
    roles_to_ne_gs = defaultdict(set)
    for i in range(len(roles_g)):
        for r in roles_g[i]:
            if r not in label_to_int_dict.keys():continue
            roles_to_ne_gs[r].add(nes_g[i])
            for children in parent_to_child_dict_gold[nes_g[i]]:
                roles_to_ne_gs[r].add(children)
    
#     print("Gold" , roles_to_ne_gs)
    roles_to_ne_pred = defaultdict(set)
    for i in range(len(nes)):
        if fl[i] == "":continue
            
        c_ri = []
        for ri in roles[i]:
            if ri in label_to_int_dict.keys():
                c_ri.append(ri)
                
        if len(c_ri) == 0:continue
        if len(ctx[i]) == 0:continue
        j = ctx[i]
        
        pred = get_labels_from_ctx_v1(nes[i], j, label_to_int_dict, model, combined, use_cache)
        
        pred = [pred]
        pred_ints = set()
        for p in pred:
            c = 0
            for p_i in p:
                if p_i == 1:
                    pred_ints.add(c)
                c = c + 1
            # pred_ints.append(torch.argmax(p))
        pred_ints = list(pred_ints)
        for ints in pred_ints:
            pred_label = int_to_label_dict[ints]
            roles_to_ne_pred[pred_label].add(nes[i])
            for children in parent_to_child_dict_sim[nes[i]]:
                roles_to_ne_pred[pred_label].add(children)
#     print("Pred", roles_to_ne_pred)
    
    label_to_jaccard = {}
    for i in roles_to_ne_gs.keys():
        gs = roles_to_ne_gs[i]
        pred = roles_to_ne_pred[i]
        label_to_jaccard[i] = jaccard_similarity(gs,pred)
    
    for i in roles_to_ne_pred.keys():
        if i in roles_to_ne_gs.keys():continue
        label_to_jaccard[i] = 0

    return (label_to_jaccard)

def get_most_occuring(l):
    c = Counter(l)
    most_common = c.most_common()
    if len(most_common) < 2:
        return most_common[0][0]
    elif most_common[0][1] > most_common[1][1]:
        return most_common[0][0]
    else:
        return ""

def evaluate_doc_v2_strict(path, label_to_int_dict, model, combined,use_cache, removed = []):
    print(path)
    
    int_to_label_dict = dict([(value, key) for key, value in label_to_int_dict.items()])
    nes_r, ctx_r, roles_r, sl_r, fl_r, variants_dict = generate_training_examples(pd.read_excel(path, index_col = 0, na_filter = None), path)
    
    roles_r_c = correct_roles(roles_r)
    
    ne_role_dict_raw = get_ne_role_dict(nes_r, roles_r_c)
    parent_to_child_dict_sim = get_parent_to_child_using_similarity(nes_r)
    
    nes, ctx, roles, sl, fl = combine_variants(nes_r, ctx_r, roles_r, sl_r, fl_r, combined, variants_dict, 0)
    nes_g, ctx_g, roles_g, sl_g, fl_g = combine_variants(nes_r, ctx_r, roles_r, sl_r, fl_r, combined, variants_dict, 1)

    parent_to_child_dict_gold = convert_ner_dict_to_parent_to_child_dict(variants_dict)
    
#     print(parent_to_child_dict_sim)
#     print(parent_to_child_dict_gold)
    
    roles = correct_roles(roles)
    roles_g = correct_roles(roles_g)
    
    text = return_text_file("./jsons/"+path[path.rfind("/")+1:-5]+".json")
    
    roles_to_ne_gs = defaultdict(set)
    for i in range(len(roles_g)):
        for r in roles_g[i]:
            if r not in label_to_int_dict.keys():continue
            roles_to_ne_gs[r].add(nes_g[i])
            for children in parent_to_child_dict_gold[nes_g[i]]:
                roles_to_ne_gs[r].add(variants_dict[children])
    
    nes_p = []
    nes_p_tags = []
    nes_p_to_tags = {}
    
    for i in range(len(nes)):
        if fl[i] == "":continue
            
        c_ri = []
        for ri in roles[i]:
            if ri in label_to_int_dict.keys():
                c_ri.append(ri)
                
        if len(c_ri) == 0:continue
        if len(ctx[i]) == 0:continue
        j = ctx[i]
        
        pred = get_labels_from_ctx_v1(nes[i], j, label_to_int_dict, model, combined, use_cache)
        
        pred = [pred]
        pred_ints = set()
        for p in pred:
            c = 0
            for p_i in p:
                if p_i == 1:
                    pred_ints.add(c)
                c = c + 1
        pred_ints = list(pred_ints)
        for ints in pred_ints:
            pred_label = int_to_label_dict[ints]
            for child in parent_to_child_dict_sim[nes[i]]:
                nes_p.append(child)
                nes_p_tags.append(pred_label)
                nes_p_to_tags[child] = pred_label
    nes_p = list(nes_p)
    nes_p_tags = list(nes_p_tags)
    
    
    roles_to_ne_pred = defaultdict(set)
    ne_parent_g_to_all_roles = defaultdict(list)
    for ne_p_i in range(len(nes_p)):
        ne_p = nes_p[ne_p_i] 
        ne_parent_g_to_all_roles[variants_dict[ne_p]].append(nes_p_tags[ne_p_i])
    
    
    for parent_ne_g in ne_parent_g_to_all_roles.keys():
        pred_label = get_most_occuring(ne_parent_g_to_all_roles[parent_ne_g])
        if pred_label == "":
            if parent_ne_g in nes_p_to_tags.keys(): 
                pred_label = nes_p_to_tags[parent_ne_g]
            else:
                continue
        roles_to_ne_pred[pred_label].add(parent_ne_g)
    
    label_to_jaccard = {}
    for i in roles_to_ne_gs.keys():
        gs = roles_to_ne_gs[i]
        pred = roles_to_ne_pred[i]
        print(i, end = " ")
        label_to_jaccard[i] = jaccard_similarity(gs,pred)
    
    for i in roles_to_ne_pred.keys():
        if i in roles_to_ne_gs.keys():continue
        label_to_jaccard[i] = 0

    return (label_to_jaccard)



from functools import reduce
  
def Average(lst):
    return reduce(lambda a, b: a + b, lst) / len(lst)

def eval_using_jaccard(model, label_to_int_dict, combined, use_cache, strict):
    train = glob.glob("./train/*.xlsx")
    test = glob.glob("./test/*.xlsx")
    labels_to_jaccards = defaultdict(list)
    c = 0
    for i in tqdm(test):
        if strict == 0:label_to_jaccard = evaluate_doc_v2(i, label_to_int_dict, model, combined, use_cache)
        if strict == 1:label_to_jaccard = evaluate_doc_v2_strict(i, label_to_int_dict, model, combined, use_cache)
        for i in label_to_jaccard.keys():
            labels_to_jaccards[i].append(label_to_jaccard[i])
        c = c + 1

    labels_to_jaccards_avg = defaultdict(list)
    for i in labels_to_jaccards.keys():
        labels_to_jaccards_avg[i] = Average(labels_to_jaccards[i])
    print(labels_to_jaccards_avg)
    return labels_to_jaccards_avg

def print_result_v2(model_name, combined, device, use_cache, cache_file, strict):
    global model_cache
    if use_cache == 2:model_cache = {}
    
    if use_cache == 1:
        with open(cache_file + '.pkl', 'rb') as f:
            model_cache = pickle.load(f)
            
    label_to_int_dict = get_label_to_int_dict()
    model = torch.load( model_name, map_location=torch.device(device))
    r = eval_using_jaccard(model, label_to_int_dict, combined, use_cache, strict)
#     print(r)
    avg_vec = []
    skip = ["NOT APPLICABLE", "OTHER"]
    for i in r.keys():
        print(i, r[i], sep = " - ")
        if i not in skip:
            avg_vec.append(r[i])
    print(sum(avg_vec)/len(avg_vec))
    
    if use_cache == 2 or use_cache == 1:
        with open(cache_file + '.pkl', 'wb') as f:
            pickle.dump(model_cache, f)
    
    return (sum(avg_vec)/len(avg_vec))