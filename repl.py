import os
import importlib
from pathlib import Path
import random
import time
import torch
import torchinfo
from tqdm import tqdm

import train
import model
import dataset

from config import DEFAULT_CONFIG, BaseConfig, ToyConfig, SmallConfig, BigConfig

config = DEFAULT_CONFIG
raw = None  # raw dataset
tok = None  # tokenizer
ds = None  # dataset tokenized and sorted
dbd = None  # dataset batched for model training
tf = None  # transformer model

random.seed(config.random_seed)
torch.manual_seed(config.random_seed)


# import any changes in the project into the repl
def rf():
    importlib.reload(train)
    importlib.reload(model)
    importlib.reload(dataset)


# initialize raw dataset
def iraw():
    global raw
    raw = dataset.get_raw_dataset(config)


# initialize tokenizer
# will retrieve from cache if tokenizer json file exists
def itok():
    global tok
    if raw is None:
        iraw()
    tok = dataset.get_tokenizer(config, raw)


# tokenize and sort dataset
def ids():
    global ds
    if tok is None:
        itok()
    ds = dataset.get_tokenized_dataset(config, tok, raw)


# initialize dynamic batched dataset
def idbd():
    global dbd
    if ds is None:
        ids()
    dbd = dataset.DynamicBatchedDataset(config, ds, tok)


# initialize transformer model
def itf():
    global tf
    if dbd is None:
        idbd(config)
    tf = model.Transformer()
    print(tf)


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
