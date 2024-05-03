import os
import importlib
from pathlib import Path
import random
import time
import torch
import torch.nn as nn
import torchinfo
from tqdm import tqdm

import components
import train
import model
import dataset

from config import DEFAULT_CONFIG, BaseConfig, ToyConfig, SmallConfig, BigConfig

config = DEFAULT_CONFIG

random.seed(config.random_seed)
torch.manual_seed(config.random_seed)


# import any changes in the project into the repl
def rf():
    global cmp
    importlib.reload(components)
    cmp = components.Components(config)
    importlib.reload(train)
    importlib.reload(model)
    importlib.reload(dataset)
    print(cmp)


cmp = components.Components(config)
print(cmp)


# sample forward pass for the model
def fp(idx=0):
    if dbd is None:
        idbd()

    transformer = model.Transformer(config)

    input_data = [
        dbd[idx]["encoder_input"],
        dbd[idx]["decoder_input"],
        dbd[idx]["source_mask"],
        dbd[idx]["target_mask"],
    ]

    print(torchinfo.summary(transformer, input_data=input_data, dtypes=[torch.int32]))


# train model
def tm():
    train.train_model(cmp)
