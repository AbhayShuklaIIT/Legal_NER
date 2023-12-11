import glob
from utility import *
from gen_train_data import *
from train import *
from validate import *
seed_everything(42)

def train_and_test(config):
    print(config)
    print('train files - ' , len(config['train']), config['train'])
    print('test files - ' , len(config['test']), config['test'])
    split_train_test(config['train'], config['test'])
    gen_training_data_cls(config['outpur_data_file'], combine_ctx = config['combined'])
    train_model(data_path = config['outpur_data_file'] + ".pkl", save_model_name = config['model_name']
          , epochs = config['epochs'], use_cuda = 1, batch_size = 16, model_load = config['model_load'])
    p,r,f = print_result_v1(config['model_name'], config['combined'], "cuda",2, config['model_name'])
    j = print_result_v2(config['model_name'], config['combined'], "cuda",1,config['model_name'],1)
    print('Results',p,r,f,j)

def perform_cross_val():
    all_files = glob.glob("./all_files/*.xlsx")
    l = int(len(all_files)/5)
    config1 = {
        'train' : all_files[l:],
        'test' : all_files[:l],
        'outpur_data_file' : 'step1_cls_c',
        'model_name' : 'step1_cls_c.pt',
        'combined' : 1,
        'model_load': '',
        'epochs' : 10
    }
    
    config2 = {
        'train' : all_files[:4*l],
        'test' : all_files[4*l:],
        'outpur_data_file' : 'step2_cls_c',
        'model_name' : 'step2_cls_c.pt',
        'combined' : 1,
        'model_load': '',
        'epochs' : 10
    }
    
    config3 = {
        'train' : all_files[:3*l] + all_files[4*l:],
        'test' : all_files[3*l:4*l],
        'outpur_data_file' : 'step3_cls_c',
        'model_name' : 'step3_cls_c.pt',
        'combined' : 1,
        'model_load': '',
        'epochs' : 10
    }

    config4 = {
        'train' : all_files[:2*l] + all_files[3*l:] ,
        'test' : all_files[2*l:3*l],
        'outpur_data_file' : 'step4_cls_c',
        'model_name' : 'step4_cls_c.pt',
        'combined' : 1,
        'model_load': '',
        'epochs' : 10
    }

    config5 = {
        'train' : all_files[:l] + all_files[2*l:] ,
        'test' : all_files[l:2*l],
        'outpur_data_file' : 'step5_cls_c',
        'model_name' : 'step5_cls_c.pt',
        'combined' : 1,
        'model_load': '',
        'epochs' : 10
    }
    train_and_test(config1)
    train_and_test(config2)
    train_and_test(config3)
    train_and_test(config4)
    train_and_test(config5)
    
perform_cross_val()