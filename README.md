# Transformer Encoder Pretraining for Language Model Creation
Pretraining transformer encoder based language models with Masked Language Modeling using HuggingFace's [transformers](https://huggingface.co/docs/transformers/index) and [datasets](https://huggingface.co/docs/datasets/index) packages. The training process should be evaluated at different checkpoints in downstream tasks to select the best model.

## Create a tokenizer (if not one is present)
Fill in the ```tokenizer_config.json``` the following fields:
```json
{
  "data_path": "/path/to/text/",
  "vocabulary_size": 50350,
  "tokenizer_type": "wordPiece or BPE", 
  "save_dir": "/path/to/save/tokenizer/"
}
```
This will return the files that will describe the tokenizer according to the selected specifications.

## Process the train data
Fill in the ```data_config.json``` the following fields:
```json
{
  "data_path": "/path/to/text/",
  "tokenizer_path": "/path/to/tokenizer/",
  "sequence_length": 512,
  "part_number": 10,
  "save_folder": "/path/to/save/processed/text/"
}
```
The processing script will create [datasets](https://huggingface.co/docs/datasets/index) arrow files. It will split the dataset in a number of parts defined in the config 
to avoid slowing down due to trying to process a single very big file.

## Pretrain the model
Fill in the ```pretraining_config.json``` the following fields:
```json
{
  "data_path": "/path/to/text/",
  "model_directory": "/path/to/model/ if you want to continue from an existing model",
  "new_model_params": { 
    "attention_heads": 12,
    "hidden_layers": 12,
    "representation_size": 768,
    "dropout": 0.1,
    "attention_dropout": 0.1
  },
  "tokenizer_directory": "/path/to/tokenizer/",
  "parameters": {
    "epochs": 40,
    "learning_rate": 0.0001,
    "batch": {
      "size": 16,
      "accumulation": 16
    },
    "warmup_steps": 10000,
    "grad_norm": 0,
    "weight_decay": 0.01,
    "adam_epsilon": 0.000001,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "mlm_probability": 0.15
  },
  "output_dir": "/path/to/output/dir/",
  "final_model_dir": "/path/to/final/model",
  "save_number": 20,
  "save_steps": 1000
}
```
Where:
* the ```mode_directory``` points to an existing model to be continued pretrained
* the ```new_model_params``` define a new model and will only be used if an existing model is not used
* the ```parameters``` define the training parameters for the session, following closely the literature (BERT)
* the ```save_number``` the number of maximum checkpoints to be saved at all times
* the ```final_model_dir``` the final model of the training process

