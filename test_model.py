import os
import shutil
import torch
from torch.optim.lr_scheduler import OneCycleLR
from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Corpus
from flair.datasets import ClassificationCorpus
from extended_flair.extended_text_classifier import TextClassifier
# from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from config.run_config import path_params, model_params


# Path preparation
PATH_PREFIX = path_params['path']
DATA_FOLDER = os.path.join(PATH_PREFIX, path_params['data_path'])

MODEL_PATH = os.path.join(PATH_PREFIX, path_params['model_path'], path_params['model_name'])

if not os.path.isdir(MODEL_PATH):
    os.makedirs(MODEL_PATH)

CACHE_DIR = os.path.join(MODEL_PATH, path_params['cache_dir']) if 'cache_dir' in path_params else None

if os.path.isdir(CACHE_DIR):
    shutil.rmtree(CACHE_DIR)
os.makedirs(CACHE_DIR)

# 1. Corpus preparation
# Make sure the corpus is in the CAPPED sentence format by running prepare_data.py
corpus: Corpus = ClassificationCorpus(
    DATA_FOLDER,
    test_file='test.csv',
    dev_file='dev.csv',
    train_file='train.csv'
)

classifier = TextClassifier.load(MODEL_PATH+'/best-model.pt')

result = classifier.evaluate(
    corpus.test,
    gold_label_type='class',
    mini_batch_size=model_params["mini_batch_size"],
)

print(result)
