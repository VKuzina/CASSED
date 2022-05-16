path_params = {
    "path": '/hdd/vkuzina/',
    "cache_dir": 'huggingface',
    "standard_data_path": 'standard_data_blank/',
    "data_path": 'data_changed_blank/',
    "results_path": "results",
    "test_file": 'test.txt',  # for eval, try also: test_other, data_model_2
    "model_path": 'models',
    "model_name": "capped_10",
}

# TRAINING AND EVAL PARAMS
model_params = {
    "learning_rate": 5.0e-5,
    "mini_batch_size": 4,
    "max_epochs": 10,
    "embeddings_storage_mode": 'none',
    "weight_decay": 0.

}

processing_params = {
    "column_name_separator": '. ',
    "sample_separator": ", ",
    "sentence_end": "."
}