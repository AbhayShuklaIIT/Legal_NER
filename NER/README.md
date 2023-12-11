#Requirements

fuzz = 0.1.1    
nltk = 3.7      
numpy = 1.22.2   
pandas = 1.4.1   
pytorch = 1.10.2  
scikit-learn = 1.0.2  
spacy = 3.2.1 
tqdm = 4.63.0
transformers = 4.17.0
nerda = 1.0.0

#Usage

Add all the excel files in the all_files folder
Add all the documents jason files in the jsons folder
Change the config files in the main file according to requirements and run main.py

Config file Description
  
{
    train : list of path of train files,
    test : list of path of train files,
    outpur_data_file : output training data file,
    model_name : output model name,
    model_load: model file path if more training is required for pre-existing model else "" ,
    epochs : Total number of epochs to train
    transformer : for LegalBERT add nlpaueb/legal-bert-base-uncased and for BERT add dslim/bert-base-NER-uncased
}
