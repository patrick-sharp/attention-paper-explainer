# scratchpad for experimenting with different python functions


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
import dataset
import model
import translate
import train
import test

import pandas as pd
from torchmetrics.text import BLEUScore


def bleu():
    predicted_text = ["this is so so great"]
    expected_text = [["this is so so great", "this is fine"]]
    metric = BLEUScore()
    print(predicted_text)
    print(expected_text)
    bleu = metric.update(predicted_text, expected_text)
    bleu = metric.compute()
    print(bleu.item())


def batches(cmp):
    total_tokens = 0
    total_tokens_wasted = 0
    for b in cmp.train_batched.batch_bounds:
        blen = b["end"] - b["start"]
        batch_tokens = b["max_len"] * blen * 2
        oop = blen * (b["md"] + b["me"])
        tokens_wasted = batch_tokens - oop
        total_tokens_wasted += tokens_wasted

        total_tokens += batch_tokens
    # generally tokens wasted without different lengths is about 5%
    return total_tokens_wasted, total_tokens, total_tokens_wasted / total_tokens


def get_synth():
    df = pd.read_csv("synthetic_dataset.csv")
    return df


def ev(cmp):
    # sentence = cmp.test_raw[0]["translation"]['de']
    # ref = cmp.test_raw[0]["translation"]['en']
    sentence = "Der Mann ging zum Markt. Er kaufte Lebensmittel."
    ref = "The man went to the market. He bought groceries."
    translations = translate.translate_beam_search(cmp, sentence)
    translate.print_comparison(sentence, ref, translations)


def tok(cmp):
    sentence = cmp.test_raw[0]["translation"]["de"]
    ref = cmp.test_raw[0]["translation"]["en"]
    return cmp.tokenizer.encode(ref)
