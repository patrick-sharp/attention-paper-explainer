import os
import pickle
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE

import dataset


def save_object(obj, filename):
    with open(filename, "wb") as file:
        pickle.dump(obj, file)


def load_object(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


class Components:
    config = None
    raw_dataset = None
    tokenizer = None
    ragged_dataset = None
    batched_dataset = None
    model = None

    def __init__(self, config):
        self.config = config

        if os.path.exists(config.raw_dataset_filename):
            self.raw_dataset = load_object(config.raw_dataset_filename)
        else:
            return

        if os.path.exists(config.tokenizer_filename):
            tokenizer = Tokenizer(BPE())
            self.tokenizer = tokenizer.from_file(self.config.tokenizer_filename)
        else:
            return

        if os.path.exists(config.ragged_dataset_filename):
            self.ragged_dataset = load_object(config.ragged_dataset_filename)
        else:
            return

        if os.path.exists(config.batched_dataset_filename):
            self.batched_dataset = load_object(config.batched_dataset_filename)

    def clean(self, components=None):
        if components is None:
            Path.unlink(self.config.raw_dataset_filename, missing_ok=True)
            self.raw_dataset = None

            Path.unlink(self.config.tokenizer_filename, missing_ok=True)
            self.tokenizer = None

            Path.unlink(self.config.ragged_dataset_filename, missing_ok=True)
            self.ragged_dataset = None

            Path.unlink(self.config.batched_dataset_filename, missing_ok=True)
            self.batched_dataset = None
        else:
            for component in components:
                if component == "raw_dataset":
                    Path.unlink(self.config.raw_dataset_filename, missing_ok=True)
                    self.raw_dataset = None
                elif component == "tokenizer":
                    Path.unlink(self.config.tokenizer_filename, missing_ok=True)
                    self.tokenizer = None
                elif component == "ragged_dataset":
                    Path.unlink(self.config.ragged_dataset_filename, missing_ok=True)
                    self.ragged_dataset = None
                elif component == "batched_dataset":
                    Path.unlink(self.config.batched_dataset_filename, missing_ok=True)
                    self.batched_dataset = None

    def init_raw_dataset(self):
        self.clean(["raw_dataset", "tokenizer", "ragged_dataset", "batched_dataset"])

        self.raw_dataset = dataset.raw_dataset(self.config)
        save_object(self.raw_dataset, self.config.raw_dataset_filename)
        return self.raw_dataset

    def init_tokenizer(self):
        self.clean(["tokenizer", "ragged_dataset", "batched_dataset"])

        if self.raw_dataset is None:
            self.init_raw_dataset()
        self.tokenizer = dataset.train_tokenizer(self.config, self.raw_dataset)
        self.tokenizer.save(self.config.tokenizer_filename)
        return self.tokenizer

    def init_ragged_dataset(self):
        self.clean(["ragged_dataset", "batched_dataset"])

        if self.tokenizer is None:
            self.init_tokenizer()
        self.ragged_dataset = dataset.tokenize_dataset(
            self.config, self.raw_dataset, self.tokenizer
        )
        save_object(self.ragged_dataset, self.config.ragged_dataset_filename)
        return self.ragged_dataset

    def init_batched_dataset(self):
        self.clean(["batched_dataset"])

        if self.ragged_dataset is None:
            self.init_ragged_dataset()
        self.batched_dataset = dataset.BatchedDataset(
            self.config, self.tokenizer, self.ragged_dataset
        )
        save_object(self.batched_dataset, self.config.batched_dataset_filename)
        return self.batched_dataset

    def init_all(self):
        self.init_batched_dataset()

    def __repr__(self):
        def status(x):
            if x == None:
                # ANSI escape characters for red
                return "\033[91m {}\033[00m".format("UNINITIALIZED")
            else:
                # ANSI escape characters for green
                return "\033[92m {}\033[00m".format("INITIALIZED")

        return (
            "Components:"
            + "\n  raw_dataset:     "
            + status(self.raw_dataset)
            + "\n  tokenizer:       "
            + status(self.tokenizer)
            + "\n  ragged_dataset:  "
            + status(self.ragged_dataset)
            + "\n  batched_dataset: "
            + status(self.batched_dataset)
        )
