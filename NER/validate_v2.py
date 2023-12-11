import glob as glob
import nltk
from utility_ner import *
from gen_train_data_ner import get_labelv2, nest_sentences, return_sents_file
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from collections import Counter
import pickle

model_cache = {}

def get_NE_from_words_tags(words, tags):
    nes = []
    nes_tags = []
    for i in range(len(tags)):
        if tags[i][0] != 'B':continue
        ne = words[i]
        ne_tag = tags[i][2:]
        j = i + 1
        continue_flag = 0
        while j < len(tags) and tags[j][0] == 'I':
            if tags[j][2:] != ne_tag:
                continue_flag = 1
                break
            ne = ne + " " + words[j]
            j = j + 1
        if continue_flag == 1:continue
        nes.append(ne)
        nes_tags.append(ne_tag)
    return nes, nes_tags

def get_NE_from_sent(model, sent, use_cache):
    if use_cache == 1:
        r = model_cache[sent]
    else:
        r = model.predict_text(sent)
    if use_cache == 2:
        model_cache[sent] = r
    words = r[0][0]
    tags = r[1][0]
    return (get_NE_from_words_tags(words, tags))

def combine_predictions_for_variant(l):
    c = Counter(l)
    most_common = c.most_common()
    if len(most_common) < 2:
        return most_common[0][0]
    elif most_common[0][1] > most_common[1][1]:
        return most_common[0][0]
    else:
        return l[0] #return first element

def get_list_for_labels(path, model, label_to_int_dict, variants_dict, use_cache):
    sents = return_sents_file("./jsons/" + path[path.rfind("/")+1:-5] + ".json", label_to_int_dict)
    nes, nes_tags = [], []
    for i in range(len(sents)):
        nes_t, nes_tags_t = get_NE_from_sent(model, sents[i], use_cache)
        nes.extend(nes_t)
        nes_tags.extend(nes_tags_t)
    
    variant_to_labels_dict = defaultdict(list)
    
    for ne_i in range(len(nes)):
        ne = nes[ne_i]
        variant_to_labels_dict[ne].append(nes_tags[ne_i])
    
    variant_to_label_dict = {}
    for i in variant_to_labels_dict.keys():
        variant_to_label_dict[i] = combine_predictions_for_variant(variant_to_labels_dict[i])
    
    roles_to_ne_pred = defaultdict(set)
    
    for i in variant_to_label_dict.keys():
        roles_to_ne_pred[variant_to_label_dict[i]].add(i)

    return roles_to_ne_pred

def get_most_occuring(l):
    c = Counter(l)
    most_common = c.most_common()
    if len(most_common) < 2:
        return most_common[0][0]
    elif most_common[0][1] > most_common[1][1]:
        return most_common[0][0]
    else:
        return ""

def get_list_for_labels_strict(path, model, label_to_int_dict, variants_dict, use_cache):
    sents = return_sents_file("./jsons/" + path[path.rfind("/")+1:-5] + ".json", label_to_int_dict)
    nes, nes_tags = [], []
    for i in range(len(sents)):
        nes_t, nes_tags_t = get_NE_from_sent(model, sents[i], use_cache)
        nes.extend(nes_t)
        nes_tags.extend(nes_tags_t)
    
    variant_to_labels_dict = defaultdict(list)
    
    for ne_i in range(len(nes)):
        ne = nes[ne_i]
        variant_to_labels_dict[ne].append(nes_tags[ne_i])
    
    variant_to_label_dict = {}
    for i in variant_to_labels_dict.keys():
        variant_to_label_dict[i] = combine_predictions_for_variant(variant_to_labels_dict[i])
    
    nes_p = list(variant_to_label_dict.keys())
    nes_p_tags = list(variant_to_label_dict.values())
    nes_p_to_tags = variant_to_label_dict
    
    roles_to_ne_pred = defaultdict(set)
    
    ne_parent_g_to_all_roles = defaultdict(list)
    for ne_p_i in range(len(nes_p)):
        ne_p = nes_p[ne_p_i] 
        if ne_p in variants_dict.keys():ne_parent_g_to_all_roles[variants_dict[ne_p]].append(nes_p_tags[ne_p_i])
        else:ne_parent_g_to_all_roles[ne_p].append(nes_p_tags[ne_p_i])
    
    for parent_ne_g in ne_parent_g_to_all_roles.keys():
        pred_label = get_most_occuring(ne_parent_g_to_all_roles[parent_ne_g])
        if pred_label == "":
            if parent_ne_g in nes_p_to_tags.keys(): 
                pred_label = nes_p_to_tags[parent_ne_g]
            else:
                continue
        roles_to_ne_pred[pred_label].add(parent_ne_g)

    return roles_to_ne_pred


from collections import defaultdict
from tqdm import tqdm
def jaccard_similarity(list1, list2):
#     print("Gold-- ", list1,"\nPred-- ", list2)
    list1 = [i.upper() for i in list1]
    list2 = [i.upper() for i in list2]
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    if union == 0: return 0 
    return float(intersection) / union

def correct_set_for_varients(l, ne_varient_dict):
    l = list(l)
    r = set()
    for i in l:
        if i in ne_varient_dict.keys():
            r.add(ne_varient_dict[i])
        else:
            r.add(i)
    return r

def evaluate_doc_v2(path, label_to_int_dict, model, use_cache):
    int_to_label_dict = dict([(value, key) for key, value in label_to_int_dict.items()])

    nes_r, ctx_r, roles_r, sl_r, fl_r, variants_dict = generate_training_examples(pd.read_excel(path, index_col = 0, na_filter = None), path)
    
    text = return_text_file("./jsons/"+path[path.rfind("/")+1:-5]+".json")
    
    roles_r = correct_roles(roles_r)
    
    roles_to_ne_gs = defaultdict(set)
    for i in range(len(roles_r)):
        roles_r[i][0] = (roles_r[i][0]).replace(" ", "-")

    #GOLD standard
    for i in range(len(roles_r)):
        for r in roles_r[i]:
            if r not in label_to_int_dict.keys():continue
            roles_to_ne_gs[r].add(nes_r[i])
    
#     print("Gold" , roles_to_ne_gs)
    
    #Prediction using model
    roles_to_ne_pred = get_list_for_labels(path, model, label_to_int_dict, variants_dict, use_cache)
    
    label_to_jaccard = {}
    for i in roles_to_ne_gs.keys():
        gs = roles_to_ne_gs[i]
        pred = roles_to_ne_pred[i]
        label_to_jaccard[i] = jaccard_similarity(gs,pred)
    
    for i in roles_to_ne_pred.keys():
        if i in roles_to_ne_gs.keys():continue
        label_to_jaccard[i] = 0

    return (label_to_jaccard)

def evaluate_doc_v2_strict(path, label_to_int_dict, model, use_cache):
    int_to_label_dict = dict([(value, key) for key, value in label_to_int_dict.items()])

    nes_r, ctx_r, roles_r, sl_r, fl_r, variants_dict = generate_training_examples(pd.read_excel(path, index_col = 0, na_filter = None), path)
    
    text = return_text_file("./jsons/"+path[path.rfind("/")+1:-5]+".json")
    
    roles_r = correct_roles(roles_r)
    
    roles_to_ne_gs = defaultdict(set)
    for i in range(len(roles_r)):
        roles_r[i][0] = (roles_r[i][0]).replace(" ", "-")

    #GOLD standard
    for i in range(len(roles_r)):
        for r in roles_r[i]:
            if r not in label_to_int_dict.keys():continue
            roles_to_ne_gs[r].add(variants_dict[nes_r[i]])
    
    #Prediction using model
    roles_to_ne_pred = get_list_for_labels_strict(path, model, label_to_int_dict, variants_dict, use_cache)
    
    label_to_jaccard = {}
    for i in roles_to_ne_gs.keys():
        gs = roles_to_ne_gs[i]
        pred = roles_to_ne_pred[i]
        label_to_jaccard[i] = jaccard_similarity(gs,pred)
    
    for i in roles_to_ne_pred.keys():
        if i in roles_to_ne_gs.keys():continue
        label_to_jaccard[i] = 0

    return (label_to_jaccard)


from functools import reduce
  
def Average(lst):
    return reduce(lambda a, b: a + b, lst) / len(lst)

def eval_using_jaccard(model, label_to_int_dict, use_cache, strict):
    labels_to_jaccards = defaultdict(list)
    c = 0;
    test_files = glob.glob("./test/*.xlsx")
    for i in tqdm(test_files):
        if strict == 0:label_to_jaccard = evaluate_doc_v2(i, label_to_int_dict, model, use_cache)
        if strict == 1:label_to_jaccard = evaluate_doc_v2_strict(i, label_to_int_dict, model, use_cache)
        for i in label_to_jaccard.keys():
            labels_to_jaccards[i].append(label_to_jaccard[i])
        c = c + 1
    labels_to_jaccards_avg = defaultdict(list)
    for i in labels_to_jaccards.keys():
        labels_to_jaccards_avg[i] = Average(labels_to_jaccards[i])
    print(labels_to_jaccards_avg)
    return labels_to_jaccards_avg


def print_result_v2(model_name, device, cache_file, use_cache, strict):
    
    global model_cache
    model_cache = {}
    
    if use_cache == 1:
        with open(cache_file + '.pkl', 'rb') as f:
            model_cache = pickle.load(f)
    
    label_to_int_dict = get_label_to_int_dict()
    model = torch.load( model_name + ".pt", map_location=torch.device(device))
    r = eval_using_jaccard(model, label_to_int_dict, use_cache, strict)
    avg_vec = []
    skip = ["NOT-APPLICABLE", "OTHER"]
    for i in r.keys():
        print(i, r[i], sep = " - ")
        if i not in skip:
            avg_vec.append(r[i])
    if len(avg_vec) != 0:
        print(sum(avg_vec)/len(avg_vec))
        return (sum(avg_vec)/len(avg_vec))
    
    if use_cache == 2:
        with open(cache_file + '.pkl', 'wb') as f:
            pickle.dump(model_cache, f)
    else: return 0
