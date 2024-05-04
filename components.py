import os
import pickle
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE

import dataset
import model
import train


class Components:
    config = None
    raw_dataset = None
    tokenizer = None
    unbatched_dataset = None
    batched_dataset = None

    epoch = None
    model = None
    optimizer = None
    model_cached = False
    optimizer_cached = False

    def __init__(self, config):
        Path(config.components_folder).mkdir(parents=True, exist_ok=True)
        self.config = config

        if self.object_cached(config.raw_dataset_filename):
            self.raw_dataset = self.load_object(config.raw_dataset_filename)
        else:
            return

        if self.object_cached(config.tokenizer_filename):
            tokenizer = Tokenizer(BPE())
            self.tokenizer = tokenizer.from_file(
                self.object_path(self.config.tokenizer_filename)
            )
        else:
            return

        if self.object_cached(config.unbatched_dataset_filename):
            self.unbatched_dataset = self.load_object(config.unbatched_dataset_filename)
        else:
            return

        if self.object_cached(config.batched_dataset_filename):
            self.batched_dataset = self.load_object(config.batched_dataset_filename)
        else:
            return

        self.init_fresh_train_state()

        if self.object_cached(config.train_state_filename):
            train_state = torch.load(self.object_path(config.train_state_filename))
            self.epoch = train_state["epoch"] + 1
            self.model.load_state_dict(train_state["model_state"])
            self.optimizer.load_state_dict(train_state["optimizer_state"])
            self.model_cached = True
            self.optimizer_cached = True

    def object_path(self, filename):
        return os.path.join(self.config.components_folder, filename)

    def object_cached(self, filename):
        path = self.object_path(filename)
        return os.path.exists(path)

    def save_object(self, obj, filename):
        path = self.object_path(filename)
        with open(path, "wb") as file:
            pickle.dump(obj, file)

    def load_object(self, filename):
        path = self.object_path(filename)
        with open(path, "rb") as file:
            return pickle.load(file)

    def clean(self, components=None):
        if components is None:
            Path.unlink(self.config.raw_dataset_filename, missing_ok=True)
            self.raw_dataset = None

            Path.unlink(self.config.tokenizer_filename, missing_ok=True)
            self.tokenizer = None

            Path.unlink(self.config.unbatched_dataset_filename, missing_ok=True)
            self.unbatched_dataset = None

            Path.unlink(self.config.batched_dataset_filename, missing_ok=True)
            self.batched_dataset = None

            Path.unlink(self.config.train_state_filename, missing_ok=True)
            self.epoch = None
            self.model = None
            self.optimizer = None
        else:
            for component in components:
                if component == "raw_dataset":
                    Path.unlink(self.config.raw_dataset_filename, missing_ok=True)
                    self.raw_dataset = None
                elif component == "tokenizer":
                    Path.unlink(self.config.tokenizer_filename, missing_ok=True)
                    self.tokenizer = None
                elif component == "unbatched_dataset":
                    Path.unlink(self.config.unbatched_dataset_filename, missing_ok=True)
                    self.unbatched_dataset = None
                elif component == "batched_dataset":
                    Path.unlink(self.config.batched_dataset_filename, missing_ok=True)
                    self.batched_dataset = None
                elif component == "train_state":
                    Path.unlink(self.config.train_state_filename, missing_ok=True)
                    self.epoch = None
                    self.model = None
                    self.optimizer = None
                else:
                    print("Error: nonexistent component {}".format(component))

    def init_raw_dataset(self):
        self.clean(
            [
                "raw_dataset",
                "tokenizer",
                "unbatched_dataset",
                "batched_dataset",
                "train_state",
            ]
        )

        self.raw_dataset = dataset.raw_dataset(self.config)
        self.save_object(self.raw_dataset, self.config.raw_dataset_filename)
        return self.raw_dataset

    def init_tokenizer(self):
        self.clean(["tokenizer", "unbatched_dataset", "batched_dataset", "train_state"])

        if self.raw_dataset is None:
            self.init_raw_dataset()
        self.tokenizer = dataset.train_tokenizer(self.config, self.raw_dataset)
        self.tokenizer.save(self.object_path(self.config.tokenizer_filename))
        return self.tokenizer

    def init_unbatched_dataset(self):
        self.clean(["unbatched_dataset", "batched_dataset", "train_state"])

        if self.tokenizer is None:
            self.init_tokenizer()
        self.unbatched_dataset = dataset.tokenize_dataset(
            self.config, self.raw_dataset, self.tokenizer
        )
        self.save_object(self.unbatched_dataset, self.config.unbatched_dataset_filename)
        return self.unbatched_dataset

    def init_batched_dataset(self):
        self.clean(["batched_dataset", "train_state"])

        if self.unbatched_dataset is None:
            self.init_unbatched_dataset()
        self.batched_dataset = dataset.BatchedDataset(
            self.config, self.tokenizer, self.unbatched_dataset
        )
        self.save_object(self.batched_dataset, self.config.batched_dataset_filename)
        return self.batched_dataset

    def init_fresh_train_state(self):
        self.epoch = 0
        self.model = model.Transformer(self.config)
        self.optimizer = train.init_optimizer(self.config, self.model)

    def init_all(self):
        self.init_batched_dataset()
        self.init_fresh_train_state()

    def __repr__(self):
        def red(x):
            # ANSI escape characters for red
            return "\033[91m{}\033[00m".format(x)

        def green(x):
            # ANSI escape characters for green
            return "\033[92m{}\033[00m".format(x)

        def status(x, extra=None):
            if x == None:
                return red("UNINITIALIZED")
            else:
                if extra is not None:
                    return green("INITIALIZED ({})".format(extra))
                else:
                    return green("INITIALIZED")

        raw_dataset_status_extra = self.raw_dataset and f"len: {len(self.raw_dataset)}"
        tokenizer_status_extra = (
            self.tokenizer and f"vocab: {self.tokenizer.get_vocab_size()}"
        )
        unbatched_dataset_status_extra = (
            self.unbatched_dataset and f"len: {len(self.unbatched_dataset)}"
        )
        batched_dataset_status_extra = (
            self.batched_dataset and f"batches: {len(self.batched_dataset)}"
        )
        epoch_status_extra = self.epoch and f"epoch: {self.epoch}"
        model_status_extra = self.model and "cached" if self.model_cached else "fresh"
        optimizer_status_extra = (
            self.optimizer and "cached" if self.optimizer_cached else "fresh"
        )

        return (
            "Components:"
            + "\n  raw_dataset:       "
            + status(self.raw_dataset, raw_dataset_status_extra)
            + "\n  tokenizer:         "
            + status(self.tokenizer, tokenizer_status_extra)
            + "\n  unbatched_dataset: "
            + status(self.unbatched_dataset, unbatched_dataset_status_extra)
            + "\n  batched_dataset:   "
            + status(self.batched_dataset, batched_dataset_status_extra)
            + "\n  epoch:             "
            + status(self.epoch, epoch_status_extra)
            + "\n  model:             "
            + status(self.model, model_status_extra)
            + "\n  optimizer:         "
            + status(self.optimizer, optimizer_status_extra)
        )
