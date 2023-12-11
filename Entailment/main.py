import glob
from utility import *
from gen_train_data import *
from train import *
from validate import *

def train_and_test(config):
    print(config)
    print('train files - ' , len(config['train']), config['train'])
    print('test files - ' , len(config['test']), config['test'])
    if config['model_load'] == '':split_train_test(config['train'], config['test'])
    if config['model_load'] == '':gen_training_data_entailment(config['outpur_data_file'], combine_ctx = config['combined'])
    train_model(data_path = config['outpur_data_file'] + ".pkl", save_model_name = config['model_name']
          , epochs = config['epochs'], combined = config['combined'], use_cuda = 1, batch_size = 16, model_load = config['model_load'])
    p,r,f = print_result_v1(config['model_name'], config['combined'], "cuda",config['model_name'], 2)
    j = print_result_v2(config['model_name'], config['combined'], "cuda", config['model_name'], 1, 1)
    print('Results',p,r,f,j)

def perform_cross_val():
    all_files = glob.glob("./all_files/*")
    l = int(len(all_files)/5)
    config1 = {
        'train' : all_files[l:],
        'test' : all_files[:l],
        'outpur_data_file' : 'step1_e_c',
        'model_name' : 'step1_e_c.pt',
        'combined' : 1,
        'model_load': '',
        'epochs' : 5
    }
    
    config2 = {
        'train' : all_files[:4*l],
        'test' : all_files[4*l:],
        'outpur_data_file' : 'step2_e_c',
        'model_name' : 'step2_e_c.pt',
        'combined' : 1,
        'model_load': '',
        'epochs' : 5
    }

    config3 = {
        'train' : all_files[:3*l] + all_files[4*l:],
        'test' : all_files[3*l:4*l],
        'outpur_data_file' : 'step3_e_c',
        'model_name' : 'step3_e_c.pt',
        'combined' : 1,
        'model_load': '',
        'epochs' : 5
    }

    config4 = {
        'train' : all_files[:2*l] + all_files[3*l:] ,
        'test' : all_files[2*l:3*l],
        'outpur_data_file' : 'step4_e_c',
        'model_name' : 'step4_e_c.pt',
        'combined' : 1,
        'model_load': '',
        'epochs' : 5
    }
    
    config5 = {
        'train' : all_files[:l] + all_files[2*l:] ,
        'test' : all_files[l:2*l],
        'outpur_data_file' : 'step5_e_c',
        'model_name' : 'step5_e_c.pt',
        'combined' : 1,
        'model_load': '',
        'epochs' : 5
    }
    
    train_and_test(config1)
    train_and_test(config2)
    train_and_test(config3)
    train_and_test(config4)
    train_and_test(config5)
    
perform_cross_val()