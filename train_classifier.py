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

# 2. create the label dictionary
LABEL_TYPE = 'class'
label_dict = corpus.make_label_dictionary(LABEL_TYPE)

# 3. initialize transformer document embeddings
document_embeddings = TransformerDocumentEmbeddings(
    model='distilbert-base-uncased', fine_tune=True, fp=8
)

# 4. create the text classifier
classifier = TextClassifier(
    document_embeddings = document_embeddings,
    label_type = LABEL_TYPE,
    label_dictionary=label_dict,
    multi_label=True,
    multi_label_threshold=0.1,
    max_token = 500,
    max_sentence_parts = 4,
    default_delimiter = '.'
)

# 5. initialize trainer with AdamW optimizer
trainer = ModelTrainer(
    classifier,
    corpus,
    optimizer=torch.optim.AdamW
)

# 7. run training with fine-tuning
trainer.train(
    base_path=MODEL_PATH,
    learning_rate=model_params["learning_rate"],
    mini_batch_size=model_params["mini_batch_size"],
    max_epochs=model_params["max_epochs"],
    scheduler=OneCycleLR,
    embeddings_storage_mode=model_params["embeddings_storage_mode"],
    weight_decay=model_params["weight_decay"],
    checkpoint = True
)

# 8. evaluate and print classification report
# The next line makes the eval not use grad
document_embeddings.training = False
result = classifier.evaluate(
    corpus.test,
    gold_label_type='class',
    mini_batch_size=model_params["mini_batch_size"],
)

