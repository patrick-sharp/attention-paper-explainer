import os
import importlib
import math
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
import scratch
import toy_dataset
import bpe_tokenizer
import masking
import model
import translation
import training
import testing
import dynamic_batched_dataset


def set_component_enum():
    """This function resets the repl's component enum values. This makes sure
    that the enum values have reference equality after you refresh the repl with rf"""
    for component_type in components.ComponentType:
        globals()[component_type.name] = component_type


def repl_state():
    global config
    global cmp
    set_component_enum()
    config = configuration.DEFAULT_CONFIG
    cmp = components.Components(config)
    print(cmp)


repl_state()


def rf():
    """refresh: re-import recent changes in the project into the repl"""
    importlib.reload(configuration)
    importlib.reload(components)
    importlib.reload(scratch)
    importlib.reload(toy_dataset)
    importlib.reload(bpe_tokenizer)
    importlib.reload(masking)
    importlib.reload(model)
    importlib.reload(translation)
    importlib.reload(training)
    importlib.reload(testing)
    importlib.reload(dynamic_batched_dataset)
    repl_state()


# sample forward pass for the model
def summary():
    transformer = model.Transformer(cmp)

    input_data = [
        cmp.train_batched[0]["encoder_input"],
        cmp.train_batched[0]["decoder_input"],
        cmp.train_batched[0]["source_mask"],
        cmp.train_batched[0]["target_mask"],
    ]

    print(torchinfo.summary(transformer, input_data=input_data, dtypes=[torch.int32]))


# translate
def translate(sentence=None):
    translation.main(sentence)


# train model
def train():
    cmp.init_all()
    training.train_model(cmp)


# test model
# prints BLEU score of model on test set
# this is a number between 0.0 and 1.0
def test():
    cmp.init_all()
    bleu_score, expected, predicted = testing.test_model(cmp)
    print("BLEU score:", bleu_score)
    for e, p in zip(expected, predicted):
        print(e)
        print(p)
        print()


# plot a color mesh of the positional encodings
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


# print n sample translations for one example from the model's training.
# manually specify a different idx to see a different translation
# this will show you how the model model improved over time.
# translations will be evenly spaced across all epochs
def print_translations(idx=0, n=5):
    if cmp.translations is None or len(cmp.translations) == 0:
        print("no translations recorded")
        return

    source_text = cmp.train_batched.examples[idx]["source"]
    target_text = cmp.train_batched.examples[idx]["target"]
    print(source_text)
    print(target_text)

    length = len(cmp.translations)
    indices = [0]
    for i in range(1, n - 1):
        indices.append(i * length // n)
    indices.append(length - 1)

    def digits(x):
        return math.floor(math.log(x, 10)) + 1

    max_i = indices[-1]
    max_epoch_digits = digits(max_i)
    epoch_width = max(len("epoch"), max_epoch_digits)

    max_ppl = max([cmp.translations[i][idx]["perplexity"] for i in indices])
    max_ppl_digits = digits(max_ppl)
    max_ppl_width = max_ppl_digits + 4  # decimal point + 3 digits of precision
    ppl_width = max(len("perplexity"), max_ppl_width)

    print(f"{'epoch':{epoch_width}}, {'perplexity':{max_ppl_width}}, translation")
    for i in indices:
        translation = cmp.translations[i][idx]["translation"]
        perplexity = cmp.translations[i][idx]["perplexity"]
        print(f"{i:{epoch_width}d}, {perplexity:{ppl_width}.3f},", translation)
