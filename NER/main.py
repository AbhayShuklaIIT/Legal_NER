import glob
from gen_train_data_ner import *
from train_ner import *
from utility_ner import *
import validate_v1 as v1
import validate_v2 as v2

def train_and_test(config):
    print(config)
    print('train files - ' , len(config['train']), config['train'])
    print('test files - ' , len(config['test']), config['test'])
    if config['model_load'] == '':split_train_test(config['train'], config['test'])
    if config['model_load'] == '':get_ner_data_and_save(config['outpur_data_file'])
    train_ner(train_file = config['outpur_data_file'], model_name = config['model_name'], epochs = config['epochs'], transformer = config['transformer'])
    p,r,f = v1.print_result_v1(config['model_name'], "cuda", config['model_name'], 2)
    j2 = v2.print_result_v2(config['model_name'], "cuda",config['model_name'], 1, 1)

def perform_cross_val():
    all_files = glob.glob("./all_files/*.xlsx")
    l = int(len(all_files)/5)
    config1 = {
        'train' : all_files[l:],
        'test' : all_files[:l],
        'outpur_data_file' : 'step1_ner_legal',
        'model_name' : 'step1_ner_legal.pt',
        'model_load': '',
        'epochs' : 50,
        'transformer' :'nlpaueb/legal-bert-small-uncased'
    }

    config2 = {
        'train' : all_files[:4*l],
        'test' : all_files[4*l:],
        'outpur_data_file' : 'step2_ner_legal_t2',
        'model_name' : 'step2_ner_legal_t2',
        'model_load': '',
        'epochs' : 50,
        'transformer' :'nlpaueb/legal-bert-small-uncased'
    }
    
    config3 = {
        'train' : all_files[:3*l] + all_files[4*l:],
        'test' : all_files[3*l:4*l],
        'outpur_data_file' : 'step3_ner_legal',
        'model_name' : 'step3_ner_legal',
        'model_load': '',
        'epochs' : 50,
        'transformer' :'nlpaueb/legal-bert-small-uncased'
    }
    config4 = {
        'train' : all_files[:2*l] + all_files[3*l:] ,
        'test' : all_files[2*l:3*l],
        'outpur_data_file' : 'step4_ner_legal',
        'model_name' : 'step4_ner_legal.pt',
        'model_load': '',
        'epochs' : 50,
        'transformer' :'nlpaueb/legal-bert-small-uncased'
        
    }
    
    config5 = {
        'train' : all_files[:l] + all_files[2*l:] ,
        'test' : all_files[l:2*l],
        'outpur_data_file' : 'step5_ner_legal',
        'model_name' : 'step5_ner_legal.pt',
        'model_load': '',
        'epochs' : 50,
        'transformer' : 'nlpaueb/legal-bert-small-uncased'
    }
    
    train_and_test(config1)
    train_and_test(config3)
    train_and_test(config3)
    train_and_test(config4)
    train_and_test(config5)
    
    
perform_cross_val()