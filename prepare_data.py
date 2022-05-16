import os
import pandas as pd
from config.run_config import path_params

PATH_PREFIX = path_params['path']
STANDARD_DATA_FOLDER = os.path.join(PATH_PREFIX, path_params['standard_data_path'])
SENTENCE_DATA_FOLDER = os.path.join(PATH_PREFIX, path_params['data_path'])

dataset_types = ['train', 'test', 'dev']

for dataset_type in dataset_types:
    value_df = pd.read_csv(STANDARD_DATA_FOLDER + dataset_type + '.csv', low_memory=False)
    label_df = pd.read_csv(STANDARD_DATA_FOLDER + dataset_type + '_labels.csv')

    columns = list(value_df.columns)
    labels = label_df['label'].to_list()
    sentences = []
    for label, column in zip(labels, columns):
        sentence = ''
        for split_label in label.split(','):
            sentence += '__label__' + split_label + ' '
        sentences.append(sentence + column + '. ' + ", ".join(str(x) for x in value_df[column].to_list()))

    with open(SENTENCE_DATA_FOLDER + dataset_type + '.csv', 'w') as outfile:
        outfile.write("\n".join(sentences))