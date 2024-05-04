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

cmp = components.Components(config)
print(cmp)


# import any changes in the project into the repl
def rf():
    global cmp
    importlib.reload(dataset)
    importlib.reload(model)
    importlib.reload(train)
    importlib.reload(components)
    cmp = components.Components(config)
    print(cmp)


# sample forward pass for the model
def fp(idx=0):

    transformer = model.Transformer(config)

    input_data = [
        cmp.batched_dataset[idx]["encoder_input"],
        cmp.batched_dataset[idx]["decoder_input"],
        cmp.batched_dataset[idx]["source_mask"],
        cmp.batched_dataset[idx]["target_mask"],
    ]

    print(torchinfo.summary(transformer, input_data=input_data, dtypes=[torch.int32]))


# train model
def tm():
    train.train_model(cmp)
