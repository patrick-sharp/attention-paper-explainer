import os
import importlib
from pathlib import Path
import time
import torch

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


def print_model():
    print(model.Transformer())


def sample_forward_pass():
    pass


def rf():
    importlib.reload(train)
    importlib.reload(model)
