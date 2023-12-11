import json
import pandas as pd
import glob
import random
import nltk
import numpy as np
import re
import re, math
from collections import Counter
from thefuzz import fuzz
import random, os
import torch

nltk.download('punkt')

label_to_int_dict = {
                     'APPELLANT': 0,
                     'JUDGE': 1,
                     'APPELLANT-COUNSEL': 2,
                     'RESPONDENT-COUNSEL': 3,
                     'RESPONDENT': 4,
                     'COURT': 5,
                     'PRECEDENT': 6,
                     'AUTHORITY': 7,
                     'WITNESS': 8,
                                         
}

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_cosine(vec1, vec2):
    # print vec1, vec2
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    WORD = re.compile(r'\w+')
    return Counter(WORD.findall(text))

def get_similarity_1(a, b):
    if "HIGH COURT" in a.upper() or "HIGH COURT" in b.upper():
        return 0
    a = text_to_vector(a.strip().lower())
    b = text_to_vector(b.strip().lower())

    return get_cosine(a, b)

def get_similarity_2(a, b):
    if "HIGH COURT" in a.upper() or "HIGH COURT" in b.upper():
        return 0
    return fuzz.ratio(a.upper(), b.upper())

def remove_annotator_added_nes(n, c, r, s, f):
    n_n, c_n, r_n, s_n, f_n = [], [], [], [], []
    n_n_a, c_n_a, r_n_a, s_n_a, f_n_a = [], [], [], [], []
    for i in range(len(n)):
        if f[i]=="":
            n_n_a.append(n[i])
            c_n_a.append(c[i])
            r_n_a.append(r[i])
            s_n_a.append(s[i])
            f_n_a.append(f[i]) 
        else:
            n_n.append(n[i])
            c_n.append(c[i])
            r_n.append(r[i])
            s_n.append(s[i])
            f_n.append(f[i])
    return n_n, c_n, r_n, s_n, f_n, n_n_a, c_n_a, r_n_a, s_n_a, f_n_a
    
def combine_variants(n, c, r, s, f, combined, annotator_variant_dict, use_anno):
    thres2 = 0.85
    thres1 = 0.85
    n_n, c_n, r_n, s_n, f_n = [], [], [], [], []
    
    if use_anno == 0:
        done = []
        n, c, r, s, f, n_a, c_a, r_a, s_a, f_a = remove_annotator_added_nes(n, c, r, s, f)
        for n_i_i in range(len(n)):
            if n_i_i in done:continue
            n_i = n[n_i_i]        
            v = []
            for n_j_i in range(len(n)):
                n_j = n[n_j_i]
                if n_i_i == n_j_i:continue
                if get_similarity_1(n_i, n_j) > thres1 and get_similarity_2(n_i, n_j) > thres2*100:
                    v.append(n_j_i)
                    done.append(n_j_i)
#                     print(n_i," --ts--",n_j)
            n_n.append(n_i)
            c_i = c[n_i_i]
            if f[n_i_i]!= "":f_i = f[n_i_i]
            else:f_i = 0
            for v_i in v:
                c_i = c_i + c[v_i]
                if f[v_i]!= "":f_i = f_i + f[v_i]
            if combined == 1:
                c_i = [". ".join(c_i)]  
            
            c_n.append(c_i)
            r_n.append(r[n_i_i])
            s_n.append(s[n_i_i])
            f_n.append(f_i)

        n_n.extend(n_a)
        c_n.extend(c_a)
        r_n.extend(r_a)
        s_n.extend(s_a)
        f_n.extend(f_a)
        return n_n, c_n, r_n, s_n, f_n
    
    elif use_anno == 1:
        for n_i_i in range(len(n)):
            n_i = n[n_i_i]        
            if annotator_variant_dict[n_i] != n_i:continue
            v = []
            for n_j_i in range(len(n)):
                n_j = n[n_j_i]
                if n_i_i == n_j_i:continue
                if annotator_variant_dict[n_j] == n_i :
                    v.append(n_j_i)
#                     print(n_i," --tr--",n_j)
            n_n.append(n_i)
            c_i = c[n_i_i]
            if f[n_i_i]!= "":f_i = f[n_i_i]
            else:f_i = 0
            for v_i in v:
                c_i = c_i + c[v_i]
                if f[v_i]!= "":f_i = f_i + f[v_i]
            if combined == 1:
                c_i = [". ".join(c_i)]  
            
            c_n.append(c_i)
            r_n.append(r[n_i_i])
            s_n.append(s[n_i_i])
            f_n.append(f[n_i_i])
        return n_n, c_n, r_n, s_n, f_n


def get_ne_varient_dict(n):
    thres = 0.85
    ne_varient_dict = {}
    done = []
    for n_i in n:
        if n_i in done:continue
        ne_varient_dict[n_i] = n_i
        for n_j in n:
            if n_i == n_j:continue
            if get_similarity_1(n_i,n_j) > thres and get_similarity_2(n_i, n_j) > thres*100:
                ne_varient_dict[n_j] = n_i
                done.append(n_j)
#                 print(n_i, n_j)
        done.append(n_i)
    return ne_varient_dict
    
def get_label_to_int_dict():
    return label_to_int_dict

def return_text_file(path):
    # print(path)
    f = open(path)
    data = json.load(f)
    text = ""
    for i in data:
        for j in (data[i]):
            text = text + " ".join(data[i][j])
    # print(data)
    return text


def get_context(ne, text, a):
    prev = " ".join(text[:a].split(" ")[-100:])
    # print(prev)
    prev_sents = nltk.tokenize.sent_tokenize(prev)
    next = " ".join(text[a+len(ne):].split(" ")[:100])
    next_sents = nltk.tokenize.sent_tokenize(next)
    ctx = " ".join(prev_sents[-1:]) + "<NE>" + ne + "</NE>" + " ".join(next_sents[:1])
    # print(ctx)
    return ctx

def remove_nan(df):
  cols = df.columns
  for col in cols:
    df = df[df[col] != np.nan]
    df = df[df[col] != float('nan')]
    df = df[df[col].notna()]
  return df


def generate_training_examples(df, path, combine_ctx = 0):
    NEs = []
    contexts = []
    roles = []
    spacy_labels = []
    df.columns = [i.lower() for i in df.columns]
    cols = ['entities', 'labels', 'frequency', 'variant', 'role']
    df = df[cols]
    df = remove_nan(df)
    r_l = list(df["role"])
    s_l = list(df["labels"])
    nes_l = df["entities"]
    variant_l = df["variant"]
    f_l = list(df["frequency"])
    
    annotator_variant_dict = {}
    text = return_text_file("./jsons/" + path[path.rfind("/")+1:-5] + ".json")
    for i in range(len(nes_l)):
        try:
            ne = nes_l[i]
        except Exception as e:
            print(e)
            continue
        try:
            a_s = [m.start() for m in re.finditer(ne, text)]
        except:
            try:
                a_s = [text.find(ne)]
            except:
                print(ne)
                continue
        context = []
        for a in a_s:
            context.append(get_context(ne, text, a))
        if combine_ctx == 1:
            ctx_temp = ". ".join(context)
            context = []
            context.append(ctx_temp)
        try:
            if len(context) > 20:
                context = context[0:20]
            r_t = r_l[i]
            p_i = int(i)
            patience = 10
            while len(r_t) == 0:
                if patience == -1: break
                try:
                    p_i = int(variant_l[p_i])
                except:
                    p_i = variant_l[p_i].split(",")
                    p_i = int(p_i[0])
                    
                r_t = r_l[p_i]
                patience = patience  - 1
                
            annotator_variant_dict[nes_l[i]] = nes_l[p_i]
            contexts.append(context)
            roles.append(r_t)
            spacy_labels.append(s_l[i])
            NEs.append(nes_l[i])
            if len(r_t) == 0:
#                 print(path, ne)
                pass
        except Exception as e: 
            print("Error catch 1" , e)
            # print(context2)
            continue
        
    return NEs, contexts, roles, spacy_labels, f_l, annotator_variant_dict


def get_data(combine_ctx, test_req = 1):
    train = glob.glob("./train/*.xlsx")
    test = glob.glob("./test/*.xlsx")
    NEs = []
    contexts = []
    roles = []
    spacy_labels = []
    cnt = 0
    for i in train:
        print(cnt, end = ", ")
        print(i)
        try:
            n, c, r, s, f, annotator_variant_dict = generate_training_examples(pd.read_excel(i, index_col = 0, na_filter = None), i, combine_ctx)
            n, c, r, s, f = combine_variants(n, c, r, s, f, combine_ctx, annotator_variant_dict, use_anno = 1)
        except Exception as e:
            print(i, "ERROR", e)
            continue
        NEs.extend(n)
        contexts.extend(c)
        roles.extend(r)
        spacy_labels.extend(s)
        cnt = cnt + 1
        
    return NEs, contexts, roles, spacy_labels, [], [], [], []

oth = ["ACQ", "ACC", 'CONV', "VIC", 'PLACE', 'VICTIM', 'Victim', 'Other', 'other', 'POL', 'MEDICAL', "O", "POLICE", "NA"]
skip = [""]
jud = ["JUDGE(CC)", "JUDGE(LC)"]
def get_label(label):
    label = label.replace(" ", "")
    label = label.upper()
    if label in skip:return ""
    if label[0:4] == "PREC" or label == "STAT" or label == "STATUTE":
        return "PRECEDENT"
    if label == "A.COUNSEL" or label == 'PETITIONER':
        return "APPELLANT COUNSEL"
    if label == "ACOUNSEL":
        return "APPELLANT COUNSEL"
    if label == "R.COUNSEL":
        return "RESPONDENT COUNSEL"
    if label == "O":
        return "OTHER"
    if label == "APP":
        return "APPELLANT"
    if label == "RESP":
        return "RESPONDENT"
    if label == "AUTH" or "AUTHORITY" in label:
        return "AUTHORITY"
    if label in ["D.WITNESS", "P.WITNESS"] or "WITNESS" in label:
        return "WITNESS"
    if label in jud:
        return "JUDGE"
    if "JUDGE" in label:
        return "JUDGE"
    if "COURT" in label:
        return "COURT"
    if "COUNSEL" in label and "APPELLANT" in label:
        return "APPELLANT COUNSEL"
    if "COUNSEL" in label and "RESPONDENT" in label:
        return "RESPONDENT COUNSEL"
    if "APPELLANT" in label:
        return "APPELLANT"
    if "RESPONDENT" in label:
        return "RESPONDENT"
    if "COUNSEL" in label:
        return "APPELLANT COUNSEL"
    for i in oth:
        if i in label:
            return "OTHER"
    return label

def correct_roles(roles):
    c_roles = []
    for rs in roles:
        try:
            crs = []
            for r in rs.split(","):
                # print(r)
                v = get_label(r)
                if v.replace(" ", "") == "" or v == "OTHER":continue
                crs.append(v)
                break
            if len(crs) == 0 and len(rs.split(",")) != 0:crs.append("OTHER")
            c_roles.append(crs)
        except Exception as e:
            print(e)
            continue
    return c_roles

def get_x_only_c(ne, c):
    return c
def get_x_c_s(ne, c):
    return ne + "[SEP]" + c

seed_everything(42)