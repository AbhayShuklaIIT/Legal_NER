Legal Role Identification (LRI) System
================================================

Overview
--------

This repository contains the implementation of a Legal Role Identification (LRI) system as described in the research paper. The system identifies specific legal roles (like judges, counsels, appellants, etc.) from legal case documents using different approaches:

1. Direct NER-based approach
2. Pipeline-based Classification approach
3. Pipeline-based Entailment approach

Key Features
------------

* Identifies 9 key legal roles:
	+ Appellant (APP)
	+ Respondent (RES)
	+ Appellant Counsel (C-APP)
	+ Respondent Counsel (C-RES)
	+ Judge (JUD)
	+ Witness (WIT)
	+ Court (CRT)
	+ Authority (AUTH)
	+ Precedent (PRE)
* Handles multiple variants of the same name
* Processes long legal documents effectively
* Provides both instance-level and document-level evaluation metrics

Requirements
------------

* Python 3.6+
* fuzz==0.1.1
* nltk==3.7
* numpy==1.22.2
* pandas==1.4.1
* pytorch==1.10.2
* scikit-learn==1.0.2
* spacy==3.2.1
* tqdm==4.63.0
* transformers==4.17.0
* nerda==1.0.0 (only for NER approach)

Project Structure
-----------------

The repository contains three main approaches in separate directories:

* NER/ - Direct NER-based approach
* CLS/ - Pipeline Classification approach
* Entailment/ - Pipeline Entailment approach

Each directory contains similar files:

* main.py - Main execution script
* utility.py - Common utility functions
* gen_train_data.py - Data preparation scripts
* train.py - Model training code
* validate.py - Evaluation scripts

Usage
-----

### Data Preparation:

* Add Excel files containing annotations to all_files/ folder
* Add document JSON files to jsons/ folder

### Configuration:

Create a config dictionary with:

   {
       "train": [list of training file paths],
       "test": [list of test file paths], 
       "output_data_file": "output training data filename",
       "model_name": "output model name",
       "combined": 1 for CAT, 0 for AGG,
       "model_load": "model path for continued training or empty string",
       "epochs": number of training epochs,
       "transformer": "transformer model name" (only for NER approach)
   }

### Run Training:

   python main.py

### Evaluation Metrics

The system provides two types of evaluation:

* Instance-level metrics (PRF):
	+ Evaluates each occurrence of named entities independently
	+ Similar to standard NER evaluation
* Document-level metrics (Jaccard):
	+ Evaluates each named entity only once per document
	+ Accounts for name variants
	+ More suitable for the LRI task

