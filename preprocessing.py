from beir import util
from beir.datasets.data_loader import GenericDataLoader
import pathlib, os, string
from tqdm import tqdm

import spacy

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import string

import collections

def download_dataset(dataset):
  '''
  PURPOSE: download the dataset
  ARGUMENSTS:
    - dataset: string describing the beir dataset
  RETURN: documents, queries, qrels of the respective dataset
  '''
  data_path = f'{dataset}'
  url = f'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip'
  out_dir = os.path.join(os.getcwd(), 'datasets')
  data_path = util.download_and_unzip(url, out_dir)
  print(f'Dataset downloaded here: {data_path}')
  return GenericDataLoader(data_path).load(split="test")

            #light        medium      heavy
#datasets = ['scifact', 'trec-covid', 'fever'] # final list of datasets
#datasets = ['scifact', 'nfcorpus'] # debugging dataset (light weight)
datasets = ['scifact', 'nfcorpus']

datasets_data = {}
number_k_prime_values = 10

for dataset in datasets:
  corpus, queries, qrels = download_dataset(dataset)
  datasets_data[dataset] = {
      'corpus': corpus,
      'queries': queries,
      'qrels': qrels
  }

k_prime_values = {dataset: [i for i in range(1, (len(data['corpus']) + 1), (len(data['corpus']) + 1)//number_k_prime_values)] for dataset, data in datasets_data.items()}
k_ground_truth = {dataset: [1, 10, 100, 1000] for dataset in datasets}

nlp = spacy.load('en_core_web_sm')
stopwords = nlp.Defaults.stop_words
clean_tokens = lambda tokens : ' '.join([token.lemma_.lower() for token in tokens if token not in stopwords and not token.is_punct])


def pre_process(elem_to_preprocess):

  key, val = elem_to_preprocess
  if type(val) is dict: # Is a document
    return key, {
        'title':  clean_tokens(nlp(val['title'])),
        'text': clean_tokens(nlp(val['text'])) # Cleaning the text document
    }
  else: return key, clean_tokens(nlp(val)) # Cleaning the query text


def query_documents_preprocessing(dataset_name, documents, queries):

  new_queries = collections.defaultdict(lambda : collections.defaultdict(dict))
  new_documents = collections.defaultdict(lambda : collections.defaultdict(dict))

  for text, iter, res in zip(('Documents', 'Queries'), (documents, queries), (new_documents, new_queries)):
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
      for id, query_doc in list(tqdm(executor.map(pre_process, iter.items()),
                                     total=len(iter.items()), desc=f'{dataset_name} - {text} Pre-Processing')):
        res[id] = query_doc

  return new_documents, new_queries


pre_processed_data = {} # Dictionary of dataset: pre_processed_corpus and pre_processed_queries
path_datasets = os.path.join(os.getcwd(), 'datasets')

for dataset, values in datasets_data.items():

  pre_proc_corpus, pre_proc_queries = None, None

  pre_proc_corpus, pre_proc_queries = query_documents_preprocessing(dataset, values['corpus'], values['queries'])
  
  pre_processed_data[dataset] = { # Populate the dictionary
      'pre_processed_corpus': pre_proc_corpus,
      'pre_processed_queries': pre_proc_queries
  }

for dataset, pre_proc_d_q in pre_processed_data.items():
    for key_name, pre_proc_val in pre_proc_d_q.items():
        pd.DataFrame.from_dict(pre_proc_val, orient='index').to_parquet(os.path.join(os.getcwd(), f'datasets/{dataset}',f'{key_name}.parquet'), index=True)
