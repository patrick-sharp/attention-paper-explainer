import os
import importlib
import math
from pathlib import Path
import random
import sys
import time
import itertools

import torch
import torch.nn as nn
import torchinfo
from tqdm import tqdm

module_strs = [
    "masking",
    "toy_dataset",
    "bpe_tokenizer",
    "model",
    "translation",
    "training",
    "testing",
    "wmt14_uniform_batch",
    "wmt14_dynamic_batch",
    "configuration",
    "component_enum",
    "components",
    "scratch",
    "plot",
]


# extremely hacky code, but this helps avoid throwing an exception when you first load
# the repl with errors in one of the imported modules.
# with this, it will just print the error and allow you to keep using the repl
def import_modules():
    for m_str in module_strs:
        m = importlib.import_module(m_str)
        importlib.reload(m)
        globals()[m_str] = m


def set_component_enum():
    """This function resets the repl's component enum values. This makes sure
    that the enum values have reference equality after you refresh the repl with rf"""
    for ct in component_enum.ComponentType:
        globals()[ct.name] = ct


def repl_state(rng_state=None):
    global config
    global cmp
    set_component_enum()
    config = configuration.CONFIG
    cmp = components.Components(config)
    print(cmp)


def rl():
    """reload: re-import module code into the repl"""
    import_modules()
    repl_state()


def all_modules_imported():
    """returns true if all necessary modules are imported, false otherwise"""
    ret_val = True
    for m_str in module_strs:
        if m_str not in globals():
            print(red("module " + m_str + " not imported"))
            ret_val = False
    return ret_val


# sample forward pass for the model
def summary():
    if not all_modules_imported():
        return

    cmp.require(TRAIN_BATCHED)

    transformer = model.Transformer(cmp)

    input_data = [
        cmp.train_batched[0]["encoder_input"],
        cmp.train_batched[0]["decoder_input"],
        cmp.train_batched[0]["source_mask"],
        cmp.train_batched[0]["target_mask"],
    ]

    print(torchinfo.summary(transformer, input_data=input_data, dtypes=[torch.int32]))


# translate a sentence using the model
def translate(sentence=None):
    if not all_modules_imported():
        return

    translation.translate(cmp, sentence)


# train model
def train(fresh=True):
    if not all_modules_imported():
        return

    cmp.require(TOKENIZER)
    cmp.require(TRAIN_BATCHED)
    if fresh:
        # reset random seeds to make training from the repl consistent as long as you're
        # training a fresh model
        random.seed(0)
        torch.manual_seed(0)
        cmp.create(MODEL_TRAIN_STATE)
    else:
        cmp.require(MODEL_TRAIN_STATE)

    training.train_model(cmp)


def time_func(func, *args, **kwargs):
    start = time.time()
    ret_val = func(*args, **kwargs)
    end = time.time()

    print(f"\nElapsed: {end-start:0.8f} seconds")

    return ret_val


# test model
# prints BLEU score of model on test set
# this is a number between 0.0 and 1.0
# set limit to None for all results
def test(limit=3):
    if not all_modules_imported():
        return

    cmp.require_all()
    bleu_score, expected, predicted = testing.test_model(cmp)
    print("BLEU score:", bleu_score)
    for e, p in itertools.islice(zip(expected, predicted), limit):
        print("expected: ", e)
        print("predicted:", p)
        print()


# print n sample translations for one example from the model's training.
# manually specify a different idx to see a different translation
# this will show you how the model model improved over time.
# translations will be evenly spaced across all epochs
def print_translations(idx=0, n=5):
    if not all_modules_imported():
        return

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


# this kicks off importing all the modules and initializing the repl state
# I put it at the end so all the other functions get defined first even if the imports have an error
rl()
