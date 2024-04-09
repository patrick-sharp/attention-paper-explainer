import os
import importlib
from pathlib import Path
import time
import torch
import torchinfo

import train
import model
import dataset


from config import DEFAULT_CONFIG, BaseConfig, ToyConfig, SmallConfig, BigConfig


def get_tokenizer(config=DEFAULT_CONFIG):
    if os.path.exists(config.tokenizer_path):
        tokenizer = dataset.load_tokenizer(config)
    else:
        tokenizer = dataset.train_tokenizer(config)
    return tokenizer


def print_model(config=DEFAULT_CONFIG):
    print(model.Transformer(config))


def sample_forward_pass(config=DEFAULT_CONFIG):
    transformer = model.Transformer(config)

    input_size = (2, config.sequence_length)

    print(torchinfo.summary(transformer, input_size=input_size, dtypes=[torch.int32]))


def rf():
    importlib.reload(train)
    importlib.reload(model)
