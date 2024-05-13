import os
import importlib
from pathlib import Path
import random
import time
import torch
import torch.nn as nn
import torchinfo
from tqdm import tqdm
import matplotlib.pyplot as plt

import components
import train
import dataset
import model
import translate

from config import DEFAULT_CONFIG

config = DEFAULT_CONFIG


def set_component_enum():
    """This function resets the repl's component enum values. This makes sure
    that the enum values have reference equality across refreshes"""
    global RAW_DATASET
    global TOKENIZER
    global UNBATCHED_DATASET
    global BATCHED_DATASET
    global MODEL_TRAIN_STATE
    RAW_DATASET = components.RAW_DATASET
    TOKENIZER = components.TOKENIZER
    UNBATCHED_DATASET = components.UNBATCHED_DATASET
    BATCHED_DATASET = components.BATCHED_DATASET
    MODEL_TRAIN_STATE = components.MODEL_TRAIN_STATE


def rf():
    """refresh: re-import recent changes in the project into the repl"""
    global cmp
    importlib.reload(dataset)
    importlib.reload(model)
    importlib.reload(train)
    importlib.reload(components)
    importlib.reload(translate)
    set_component_enum()
    cmp = components.Components(config)
    print(cmp)


set_component_enum()
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


# translate
def tr():
    translate.main([None])


# train model
def tm():
    cmp.init_all()
    train.train_model(cmp)


# plot a graph of the current loss
def plot_loss():
    plt.plot(cmp.losses)
    plt.ylabel("Loss")
    plt.xlabel("Batch")
    ax = plt.gca()
    ax.set_ylim([0, cmp.losses[0] + 1])
    plt.show()


def plot_positional_encodings():
    if not cmp.present(MODEL_TRAIN_STATE):
        cmp.create(MODEL_TRAIN_STATE)

    tokens = 10

    # (10, d_model)
    pos_encoding = cmp.model.positional_encoding.positional_encodings[0, 0:tokens, :]

    plt.figure(figsize=(12, 8))
    plt.pcolormesh(pos_encoding, cmap="viridis")
    plt.xlabel("Embedding Dimensions")
    plt.xlim((0, cmp.config.d_model))
    plt.ylim((tokens, 0))
    plt.ylabel("Token Position")
    plt.colorbar()
    plt.show()
