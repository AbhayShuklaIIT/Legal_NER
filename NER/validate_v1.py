import glob as glob
import nltk
from utility_ner import *
from gen_train_data_ner import get_labelv2, nest_sentences, return_sents_file
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from collections import defaultdict
from tqdm import tqdm
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


def get_list_for_labels(path, model, label_to_int_dict, variants_dict, use_cache):
    sents = return_sents_file("./jsons/" + path[path.rfind("/")+1:-5] + ".json", label_to_int_dict)
    nes, nes_tags = [], []
    for i in range(len(sents)):
        nes_t, nes_tags_t = get_NE_from_sent(model, sents[i], use_cache)
        nes.extend(nes_t)
        nes_tags.extend(nes_tags_t)
    roles_to_ne_pred = defaultdict(list)
    for i in range(len(nes_tags)):
        tag = nes_tags[i]
        if nes[i] in variants_dict.keys():
            p_ne = variants_dict[nes[i]]
        else:
            p_ne = nes[i]
        roles_to_ne_pred[tag].append(p_ne)
    return roles_to_ne_pred


def convert_ne_and_tags_to_y(nes, tags,label_to_int_dict):
    ne_to_tags = defaultdict(set)
    for i in range(len(nes)):
        ne_to_tags[nes[i]].add(label_to_int_dict[tags[i]])
    print(ne_to_tags)


def intger_to_one_hot(h, label_to_int_dict):
    result = []
    for i in range(len(label_to_int_dict.keys())):
        if i == h:
            result.append(1)
        else:
            result.append(0)
    return result

def correct_list_for_varients(l, variants_dict):
    r = []
    for i in l:
        if i in variants_dict.keys():
            r.append(variants_dict[i])
        else:
            r.append(i)
    return r

def find_indices(lst, element):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(element, offset+1)
        except ValueError:
            return result
        result.append(offset)

def get_ne_to_freq(nes, fl):
    ne_to_freq = {}
    for i in range(len(nes)):
        ne_to_freq[nes[i]] = fl[i]
    return ne_to_freq

def convert_variants_dict_to_parent_to_child_dict(variants_dict):
    parent_to_child_dict = defaultdict(list)
    for i in variants_dict.keys():
        parent_to_child_dict[variants_dict[i]].append(i)
    return parent_to_child_dict

def get_freq(ne_g, parent_to_child_dict_g, ne_to_freq_r, text):
    f = 0
    for child in parent_to_child_dict_g[ne_g]:
        if ne_to_freq_r[child] != "":f = f + ne_to_freq_r[child]
        else: f = f + text.count(ne_g)
    return f

def evaluate_doc_v1(path, label_to_int_dict, model, use_cache):
    int_to_label_dict = dict([(value, key) for key, value in label_to_int_dict.items()])
    text = return_text_file("./jsons/"+path[path.rfind("/")+1:-5]+".json")
    
    nes_r, ctx_r, roles_r, sl_r, fl_r, variants_dict = generate_training_examples(pd.read_excel(path, index_col = 0, na_filter = None), path)
    nes_g, ctx, roles_g, sl, fl = combine_variants(nes_r, ctx_r, roles_r, sl_r, fl_r, 0, variants_dict, 1)
    
    ne_to_freq_r = get_ne_to_freq(nes_r, fl_r)
    parent_to_child_dict_g = convert_variants_dict_to_parent_to_child_dict(variants_dict)
    
    roles_g = correct_roles(roles_g)
    for i in range(len(roles_g)):
        roles_g[i][0] = (roles_g[i][0]).replace(" ", "-")
    #Prediction using model
    sents = return_sents_file("./jsons/" + path[path.rfind("/")+1:-5] + ".json", label_to_int_dict)
    nes_p, nes_tags_p = [], []
    for i in range(len(sents)):
        nes_t, nes_tags_t = get_NE_from_sent(model, sents[i], use_cache)
        nes_p.extend(nes_t)
        nes_tags_p.extend(nes_tags_t)
    nes_p = correct_list_for_varients(nes_p, variants_dict)
    yhat = []
    ytest = []
    for ne_g_i in range(len(nes_g)):
        ne_g = nes_g[ne_g_i]
        if len(roles_g[ne_g_i])==0:continue
        if roles_g[ne_g_i][0] == "NOT-APPLICABLE": continue
        if roles_g[ne_g_i][0] not in label_to_int_dict.keys(): continue
        freq = get_freq(ne_g, parent_to_child_dict_g, ne_to_freq_r,text)
        if ne_g in nes_p:
            indices = find_indices(nes_p, ne_g)
            yt = intger_to_one_hot(label_to_int_dict[roles_g[ne_g_i][0]], label_to_int_dict)
            f = max(freq,len(indices))
            ytest.extend([yt]*f)
            for j in indices:
                yh = intger_to_one_hot(label_to_int_dict[nes_tags_p[j]], label_to_int_dict)
                yhat.append(yh)
            if freq - len(indices) > 0:
                for i in range(freq - len(indices)):
                    yh = [0 for _ in range(len(label_to_int_dict))]
                    yhat.append(yh)
        else:
            yt = intger_to_one_hot(label_to_int_dict[roles_g[ne_g_i][0]], label_to_int_dict)
            ytest.extend([yt]*freq)
            yh = [0 for _ in range(len(label_to_int_dict))]
            yhat.extend([yh]*freq)
            
    for ne_p_i in range(len(nes_p)):
        ne_p = nes_p[ne_p_i]
        if ne_p not in nes_g:
            yt = [0 for _ in range(len(label_to_int_dict))]
            ytest.append(yt)
            yh = intger_to_one_hot(label_to_int_dict[nes_tags_p[ne_p_i]], label_to_int_dict)
            yhat.append(yh)
    return ytest, yhat


def evaluate_v1(label_to_int_dict, model, use_cache):
    yhat = []
    ytest = []
    test_files = glob.glob("./test/*.xlsx")
    for i in tqdm(test_files):
        print(i)
        yt, yh = evaluate_doc_v1(i, label_to_int_dict, model, use_cache)
        
        if len(yt) != len(yh):
            print(i, "not sam len")
            continue
        ytest.extend(yt)
        yhat.extend(yh)
    return ytest, yhat


def del_labels(ytest, yhat, label_to_int_dict):
    return ytest, yhat, list(label_to_int_dict.keys())


def print_cls_report_and_vals(ytest_f, yhat_f, labels, thres = 0.5):
    col = len(labels) +4
    yhat_ct = np.array(yhat_f) > thres
    print(classification_report(np.array(ytest_f), yhat_ct, target_names = labels, digits = 4))
    f1 = (classification_report(np.array(ytest_f), yhat_ct, target_names = labels).split("\n")[col].split("      ")[4])
    p = (classification_report(np.array(ytest_f), yhat_ct, target_names = labels).split("\n")[col].split("      ")[2])
    r = (classification_report(np.array(ytest_f), yhat_ct, target_names = labels).split("\n")[col].split("      ")[3])
    return p, r, f1


def print_result_v1(model_name, device, cache_file, use_cache):
    global model_cache
    model_cache = {}
    
    if use_cache == 1:
        with open(cache_file + '.pkl', 'rb') as f:
            model_cache = pickle.load(f)
    
    label_to_int_dict = get_label_to_int_dict()
    model = torch.load( model_name + ".pt", map_location=torch.device(device)) 
    ytest, yhat = evaluate_v1(label_to_int_dict, model, use_cache)
    ytest_n, yhat_n, labels = del_labels(ytest, yhat, label_to_int_dict)
    p, r, f = print_cls_report_and_vals(ytest_n, yhat_n, labels)
    
    if use_cache == 2:
        with open(cache_file + '.pkl', 'wb') as f:
            pickle.dump(model_cache, f)
    
    return p, r, f