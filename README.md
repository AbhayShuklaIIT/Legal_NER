# Legal Role Identification (LRI) System

## Overview
This repository implements a Legal Role Identification (LRI) system as detailed in the associated research paper. The system identifies specific legal roles (such as judges, counsels, appellants, etc.) from legal case documents using various approaches:
1. Direct NER-based approach
2. Pipeline-based Classification approach
3. Pipeline-based Entailment approach

## Key Features
- Identifies 9 key legal roles:
  - Appellant (APP)
  - Respondent (RES)
  - Appellant Counsel (C-APP)
  - Respondent Counsel (C-RES)
  - Judge (JUD)
  - Witness (WIT)
  - Court (CRT)
  - Authority (AUTH)
  - Precedent (PRE)
- Handles multiple variants of the same name
- Effectively processes long legal documents
- Provides both instance-level and document-level evaluation metrics

## Requirements
- Python 3.6+
- fuzz==0.1.1
- nltk==3.7
- numpy==1.22.2
- pandas==1.4.1
- pytorch==1.10.2
- scikit-learn==1.0.2
- spacy==3.2.1
- tqdm==4.63.0
- transformers==4.17.0
- nerda==1.0.0 (only for NER approach)

## Project Structure
The repository contains three main approaches in separate directories:
- NER/ - Direct NER-based approach
- CLS/ - Pipeline Classification approach
- Entailment/ - Pipeline Entailment approach

Each directory includes similar files:
- `main.py` - Main execution script
- `utility.py` - Common utility functions
- `gen_train_data.py` - Data preparation scripts
- `train.py` - Model training code
- `validate.py` - Evaluation scripts

## Usage
### Data Preparation:
- Add Excel files containing annotations to the `all_files/` folder
- Add document JSON files to the `jsons/` folder

### Configuration:
Create a config dictionary with:
