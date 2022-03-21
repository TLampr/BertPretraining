from transformers import RobertaTokenizerFast, BertTokenizerFast, AutoTokenizer

from datasets import load_dataset

import argparse
from pathlib import Path
import json


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // LEN) * LEN  # 512
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + LEN] for i in range(0, total_length, LEN)]
        for k, t in concatenated_examples.items()
    }
    return result


def special_tokens_mask(examples):
    length = len(examples['input_ids'])
    return [0] * (length - 1) + [1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-config_dir', dest='config', type=str,
                        help='Path to the configuration json file', required=False, default="./data_config.json")
    args = parser.parse_args()

    print("\n" + f"USING {args.config} AS A CONFIG FILE")

    config_file = open(args.config)
    config = json.load(config_file)

    print('\n' + json.dumps(config, indent=4, sort_keys=True))

    LEN = config["sequence_length"] - 2

    # will return the token ids and the attention masks
    def encode_function(examples):
        examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
        return tokenizer(
            examples["text"],
            add_special_tokens=False,
            return_token_type_ids=False,
            return_special_tokens_mask=True
        )

    try:
        # If you provide the config of the model in the same folder it will load the proper tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
    except:
        # If you don't you need to specify it
        try:
            tokenizer = RobertaTokenizerFast.from_pretrained(config["tokenizer_path"])
        except:
            tokenizer = BertTokenizerFast.from_pretrained(config["tokenizer_path"])

    training_paths = [str(x) for x in Path(config["data_path"]).glob("*.txt")]
    datasets = load_dataset(
        'text',
        data_files=training_paths,
        split=[f'train[{k}%:{k+config["part_number"]}%]' for k in range(0, 100, config["part_number"])],
        cache_dir=args.data + '/cache'
    )
    for index, dataset in enumerate(datasets):
        print(f'PART {index + 1}')
        # tokenize the initial blocks of text
        dataset = dataset.map(encode_function, batched=True, remove_columns=['text'])
        # add the separation </s> token at the end of each original row/sample/entry of text
        # according to the RoBERTA paper (separate different documents)
        dataset = dataset.map(
            lambda example: {'input_ids': example['input_ids'] + [tokenizer.sep_token_id],
                             'attention_mask': example['attention_mask'] + [1],
                             'special_tokens_mask': example['special_tokens_mask'] + [1]}
        )
        # concatenate the tokenized blocks of text to 510 blocks leaving 2 token spots for the <s> and </s> tokens
        dataset = dataset.map(group_texts, batched=True)
        # add the <s> token at the beginning and the </s> token at the end of each of the new blocks
        # and adjust the attention mask and special tokens mask accordingly
        dataset = dataset.map(
            lambda example: {'input_ids': [tokenizer.cls_token_id] + example['input_ids'] + [tokenizer.sep_token_id],
                             'attention_mask': example['attention_mask'] + [1, 1],
                             'special_tokens_mask': [1] + example['special_tokens_mask'] + [1]}
        )
        processed_dataset_path = config["save_folder"] + '/processed' + f'/part_{index + 1}'
        dataset.save_to_disk(processed_dataset_path)
        dataset.cleanup_cache_files()
