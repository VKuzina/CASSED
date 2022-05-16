**CAPPED**

Capped is a model for the detection of sensitive data in structured datasets, more specificly, for the multilabel problem of columns in database tables.

The model uses the BERT model#, through the Flair library#, and has an accompanying dataset on kaggle called DeSSI (Dataset for Structured Sensitive Information)#.

To learn more about the model please refer to the full paper ##.

**Setup**
All of the setup for CAPPED is made in configs/run_config.py, where inside of the path_params dictionary, all of the paths need to be set.

**Datasets**
Several datasets are present in the repository and can be found in ##, set the path to the dataset in the run_config.py. If a different dataset is desired, for CAPPED to work on it, the dataset needs to be in a specific format. To turn the standard format of .csv data and labels, readable into a pandas Dataframe, into CAPPED-s required format, you can simply set the parameter "standard_data_path" inside of the path_parameters to the folder with the standardised data, and run prepare_data.py.

**Use trained models**
A wide variets of models are pretrained on different datasets, and are available in the repository under ##. Set the path parameters in the run_config.py file to the desired path, and run test_model.py.

**Train your own model**
To train your own model, set the path to the dataset in the path_parameters inside the run_config.py file, as well as the model_parameters and the processing_parameters to your desired values, prepare the data, as described before in the Datasets section, and run train_classifier.py
