from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel, Whitespace
from tokenizers.trainers import BpeTrainer, WordPieceTrainer

from pathlib import Path
import argparse
import os
import json


# This trains a new tokenizer
def train_new_tokenizer(training_text_dir: str = ".", vocab_size: int = 35_000, save_dir: str = "./",
                        tokenizer_type: str = 'BPE'):
    if tokenizer_type.lower() == 'bpe':
        print(f'Create a {tokenizer_type} tokenizer\n')
        # First create an empty Byte-Pair encoding model
        new_tokenizer = Tokenizer(BPE())
        new_tokenizer.pre_tokenizer = ByteLevel()
        new_tokenizer.decoder = ByteLevelDecoder()
        special_tokens = ['<s>', '</s>', '<unk>', '<pad>', '<mask>']
    else:
        print(f'Create a {tokenizer_type} tokenizer\n')
        new_tokenizer = Tokenizer(WordPiece())
        new_tokenizer.pre_tokenizer = Whitespace()
        new_tokenizer.decoder = WordPieceDecoder()
        special_tokens = ['[CLS]', '[SEP]', '[UNK]', '[PAD]', '[MASK]']
    # Then enable lower-casing and unicode-normalization
    new_tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
    if tokenizer_type.lower() == 'bpe':
        tokenizer_trainer = BpeTrainer(vocab_size=vocab_size,
                                       show_progress=True,
                                       initial_alphabet=ByteLevel.alphabet(),
                                       special_tokens=special_tokens
                                       )
    else:
        tokenizer_trainer = WordPieceTrainer(vocab_size=vocab_size,
                                             show_progress=True,
                                             special_tokens=special_tokens
                                             )
    paths = [str(x) for x in Path(training_text_dir).glob("*.txt")]
    new_tokenizer.train(trainer=tokenizer_trainer, files=paths)
    new_tokenizer.model.save(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-config_dir', dest='config', help='Path to the configuration json file',
                        type=str, required=False, default='./tokenizer_config.json')
    args = parser.parse_args()

    print("\n" + f"USING {args.config} AS A CONFIG FILE")
    config_file = open(args.config)
    config = json.load(config_file)

    print("\n" + json.dumps(config, indent=4, sort_keys=True))

    os.makedirs(os.path.dirname(config["save_dir"]), exist_ok=True)

    train_new_tokenizer(training_text_dir=config["data_path"],
                        vocab_size=config["vocabulary_size"],
                        save_dir=config["save_dir"],
                        tokenizer_type=config["tokenizer_type"]
                        )
