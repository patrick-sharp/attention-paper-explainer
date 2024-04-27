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

raw = None  # raw dataset
tok = None  # tokenizer
ds = None  # dataset tokenized and sorted
dbd = None  # dataset batched for model training
tf = None  # transformer model


# import any changes in the project into the repl
def rf():
    importlib.reload(train)
    importlib.reload(model)
    importlib.reload(dataset)


# initialize raw dataset
def iraw(config=DEFAULT_CONFIG):
    global raw
    raw = dataset.get_raw_dataset(config)


# initialize tokenizer
# will retrieve from cache if tokenizer json file exists
def itok(config=DEFAULT_CONFIG):
    global tok
    if raw is None:
        iraw(config)
    tok = dataset.get_tokenizer(config, raw)


# tokenize and sort dataset
def ids(config=DEFAULT_CONFIG):
    global ds
    if tok is None:
        itok(config)
    ds = dataset.get_tokenized_dataset(config, tok, raw)


# initialize dynamic batched dataset
def idbd(config=DEFAULT_CONFIG):
    global dbd
    if ds is None:
        ids(config)
    dbd = dataset.DynamicBatchedDataset(config, ds, tok)


# initialize transformer model
def itf(config=DEFAULT_CONFIG):
    global tf
    if dbd is None:
        idbd(config)
    tf = model.Transformer()
    print(tf)


# sample forward pass for the model
def fp(config=DEFAULT_CONFIG):
    transformer = model.Transformer(config)

    input_size = (2, config.sequence_length)

    print(torchinfo.summary(transformer, input_size=input_size, dtypes=[torch.int32]))


# TODO: see if you can repro EOS at the beginning of tokenized input
