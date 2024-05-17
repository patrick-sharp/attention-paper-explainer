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

import configuration
import components
import train
import dataset
import model
import translate
import test


def set_component_enum():
    """This function resets the repl's component enum values. This makes sure
    that the enum values have reference equality across refreshes"""
    for component_type in components.ComponentType:
        globals()[component_type.name] = component_type


set_component_enum()
config = configuration.DEFAULT_CONFIG
cmp = components.Components(config)
print(cmp)


def rf():
    """refresh: re-import recent changes in the project into the repl"""
    importlib.reload(configuration)
    importlib.reload(dataset)
    importlib.reload(model)
    importlib.reload(train)
    importlib.reload(components)
    importlib.reload(translate)
    importlib.reload(test)

    global config
    global cmp
    set_component_enum()
    config = configuration.DEFAULT_CONFIG
    cmp = components.Components(config)
    print(cmp)


# sample forward pass for the model
def fp(idx=0):
    transformer = model.Transformer(config)

    input_data = [
        cmp.train_batched[idx]["encoder_input"],
        cmp.train_batched[idx]["decoder_input"],
        cmp.train_batched[idx]["source_mask"],
        cmp.train_batched[idx]["target_mask"],
    ]

    print(torchinfo.summary(transformer, input_data=input_data, dtypes=[torch.int32]))


# translate
def tr():
    translate.main([None])


# train model
def tm():
    cmp.init_all()
    train.train_model(cmp)


# test (i.e. eval) model
def ev():
    cmp.init_all()
    return test.test_model(cmp)


def plot_positional_encodings():
    """This code is from Jay Alammar"""

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


# plot a graph of the current loss
def plot_loss():
    plt.plot(cmp.losses)
    plt.ylabel("Loss")
    plt.xlabel("Batch")
    ax = plt.gca()
    ax.set_ylim([0, cmp.losses[0] + 1])
    plt.show()


def print_translations(idx=0, n=5):
    if cmp.translations is None or len(cmp.translations) == 0:
        print("no translations recorded")
        return

    length = len(cmp.translations)
    indices = [0]
    for i in range(1, n - 1):
        indices.append(i * length // n)
    indices.append(length - 1)
    for i in indices:
        print()
        print(cmp.translations[i][idx]["translation"])
