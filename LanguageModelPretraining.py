from transformers import BertForMaskedLM, BertConfig, BertTokenizerFast
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer

from datasets import Dataset, concatenate_datasets

import argparse
from pathlib import Path
import re
import json
import os


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config_dir', dest='config', type=str, required=False, default='./pretraining_config.json',
                        help='Path to the configuration json file')
    args = parser.parse_args()

    print("\n" + f"USING {args.config} AS A CONFIG FILE")

    config_file = open(args.config)
    config = json.load(config_file)

    print('\n' + json.dumps(config, indent=4, sort_keys=True))

    # if we are going to use an already pre trained model
    if "model_directory" in config:
        model = AutoModelForMaskedLM.from_pretrained(config["model_directory"])
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_directory"], model_max_length=512, max_length=512)
    else:
        print('\nInitialize a new model')
        # if we are going to use a new model just leave empty the model directory
        tokenizer = BertTokenizerFast.from_pretrained(config["tokenizer_directory"],
                                                      model_max_length=512,
                                                      max_length=512
                                                      )
        model_config = BertConfig(vocab_size=tokenizer.get_vocab().__len__(),
                                  max_position_embeddings=tokenizer.model_max_length,
                                  num_attention_heads=config["new_model_params"]["attention_heads"],
                                  num_hidden_layers=config["new_model_params"]["hidden_layers"],
                                  hidden_size=config["new_model_params"]["representation_size"],
                                  hidden_dropout_prob=config["new_model_params"]["dropout"],
                                  attention_probs_dropout_prob=config["new_model_params"]["attention_dropout"]
                                  )
        model = BertForMaskedLM(config=model_config)
    # finds the different data part folders in the dataset directory
    data_parts = [str(x) for x in Path(config["data_path"] + "/processed/").glob("*")]
    data_parts.sort(key=natural_keys)
    # concatenates all the parts back into the complete dataset
    data_part_list = [Dataset.load_from_disk(dataset_path=data_part) for data_part in data_parts]
    trainDataset = concatenate_datasets(data_part_list)

    # it will apply the mask token dynamically in each sample
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=config["parameters"]["mlm_probability"]
    )

    # training arguments, closely following RoBERTa
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        overwrite_output_dir=True,
        num_train_epochs=config["parameters"]["epochs"],
        per_device_train_batch_size=config["parameters"]["batch"]["size"],
        gradient_accumulation_steps=config["parameters"]["batch"]["accumulation"],
        save_total_limit=config["save_number"],
        max_grad_norm=config["parameters"]["grad_norm"],
        load_best_model_at_end=False,
        save_steps=config["save_steps"],
        warmup_steps=config["parameters"]["warmup_steps"],
        weight_decay=config["parameters"]["weight_decay"],
        learning_rate=config["parameters"]["learning_rate"],
        adam_epsilon=config["parameters"]["adam_epsilon"],
        adam_beta1=config["parameters"]["adam_beta1"],
        adam_beta2=config["parameters"]["adam_beta2"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=trainDataset
    )

    # if you want to start from a checkpoint add the "checkpoint_dir" entry and path in the json file
    if "checkpoint_dir" in config:
        trainer.train(resume_from_checkpoint=config["checkpoint_dir"])
    else:
        trainer.train()
    os.makedirs(os.path.dirname(config["final_model_dir"]), exist_ok=True)
    trainer.save_model(config["final_model_dir"])
