import os
import importlib
from pathlib import Path
import time
import torch
import torchinfo
from tqdm import tqdm

import train
import model
import dataset


from config import DEFAULT_CONFIG, BaseConfig, ToyConfig, SmallConfig, BigConfig


def tokenize_all(config=DEFAULT_CONFIG):
    ds = dataset.get_dataset(config)
    if os.path.exists(config.tokenizer_path):
        tokenizer = dataset.load_tokenizer(config)
    else:
        tokenizer = dataset.train_tokenizer(config, ds)

    tokenized = [None] * len(ds)
    for i, item in tqdm(enumerate(ds)):
        sentence_pair = item["translation"]
        tokenized[i] = {
            "de": tokenizer.encode(sentence_pair["de"]),
            "en": tokenizer.encode(sentence_pair["en"]),
        }
    return tokenized, ds


def gimme_data(config=DEFAULT_CONFIG):
    return dataset.DynamicBatchedDataset(config)


def print_model(config=DEFAULT_CONFIG):
    print(model.Transformer(config))


def sample_forward_pass(config=DEFAULT_CONFIG):
    transformer = model.Transformer(config)

    input_size = (2, config.sequence_length)

    print(torchinfo.summary(transformer, input_size=input_size, dtypes=[torch.int32]))


def rf():
    importlib.reload(train)
    importlib.reload(model)
    importlib.reload(dataset)

# TODO: repl state for better incremental developement
# TODO: see if you can repro EOS at the beginning of tokenized input
