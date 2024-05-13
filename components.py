import os
from enum import Enum, global_enum
import pickle
from pathlib import Path
import random

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE

import dataset
import model
import train


def save_pickle(obj, path):
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


@global_enum
class ComponentType(Enum):
    RAW_DATASET = 0
    TOKENIZER = 1
    UNBATCHED_DATASET = 2
    BATCHED_DATASET = 3
    MODEL_TRAIN_STATE = 4


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
    losses = None

    def __init__(self, config):
        random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        Path(config.components_folder).mkdir(parents=True, exist_ok=True)
        self.config = config

        def join_path(filename):
            return os.path.join(config.components_folder, filename)

        # allows us to access component types with reference equality in the train loop
        # and in single-sentence translation
        self.types = ComponentType

        self.paths = {
            RAW_DATASET: join_path(config.raw_dataset_filename),
            UNBATCHED_DATASET: join_path(config.unbatched_dataset_filename),
            TOKENIZER: join_path(config.tokenizer_filename),
            BATCHED_DATASET: join_path(config.batched_dataset_filename),
            MODEL_TRAIN_STATE: join_path(config.model_train_state_filename),
        }

        self.present = {
            RAW_DATASET: False,
            UNBATCHED_DATASET: False,
            TOKENIZER: False,
            BATCHED_DATASET: False,
            MODEL_TRAIN_STATE: False,
        }

        self.load_all()

    def exists(self, component_type):
        """return whether a cached version of this component exists"""
        path = self.paths[component_type]
        return os.path.exists(path)

    def create(self, component_type):
        """create and save a component"""
        if component_type.value == RAW_DATASET.value:
            self.clean(MODEL_TRAIN_STATE)
            self.clean(BATCHED_DATASET)
            self.clean(UNBATCHED_DATASET)
            self.clean(TOKENIZER)
            self.clean(RAW_DATASET)
            component = dataset.raw_dataset(self.config)
            self.raw_dataset = component
        elif component_type == TOKENIZER:
            self.clean(MODEL_TRAIN_STATE)
            self.clean(BATCHED_DATASET)
            self.clean(UNBATCHED_DATASET)
            self.clean(TOKENIZER)
            if not self.present[RAW_DATASET]:
                self.create(RAW_DATASET)
            component = dataset.train_tokenizer(self.config, self.raw_dataset)
            self.tokenizer = component
        elif component_type == UNBATCHED_DATASET:
            self.clean(MODEL_TRAIN_STATE)
            self.clean(BATCHED_DATASET)
            self.clean(UNBATCHED_DATASET)
            if not self.present[TOKENIZER]:
                self.create(TOKENIZER)
            component = dataset.tokenize_dataset(
                self.config, self.raw_dataset, self.tokenizer
            )
            self.unbatched_dataset = component
        elif component_type == BATCHED_DATASET:
            self.clean(MODEL_TRAIN_STATE)
            self.clean(BATCHED_DATASET)
            if not self.present[UNBATCHED_DATASET]:
                self.create(UNBATCHED_DATASET)
            component = dataset.BatchedDataset(
                self.config, self.tokenizer, self.unbatched_dataset
            )
            self.batched_dataset = component
        elif component_type == MODEL_TRAIN_STATE:
            # model train state is a special case.
            # it doesn't depend on other components since we want to be able to use it
            # to translate an arbitrary sentence without loading the dataset
            self.clean(MODEL_TRAIN_STATE)
            self.epoch = 0
            self.model = model.Transformer(self.config)
            self.optimizer = train.init_optimizer(self.config, self.model)
            self.losses = []
            # don't set component; the train loop handles saving the state

        # the train loop handles initializing and saving the model state
        if component_type != MODEL_TRAIN_STATE:
            self.save(component_type, component)

        self.present[component_type] = True

    def save(self, component_type, component):
        path = self.paths[component_type]
        if component_type == RAW_DATASET:
            save_pickle(component, path)
        elif component_type == TOKENIZER:
            self.tokenizer.save(path)
        elif component_type == UNBATCHED_DATASET:
            save_pickle(component, path)
        elif component_type == BATCHED_DATASET:
            save_pickle(component, path)
        elif component_type == MODEL_TRAIN_STATE:
            torch.save(component, path)

    def load(self, component_type):
        if not self.exists(component_type):
            return

        path = self.paths[component_type]

        if component_type == RAW_DATASET:
            self.raw_dataset = load_pickle(path)
        elif component_type == TOKENIZER:
            tokenizer = Tokenizer(BPE())
            tokenizer = tokenizer.from_file(path)
            self.tokenizer = tokenizer
        elif component_type == UNBATCHED_DATASET:
            self.unbatched_dataset = load_pickle(path)
        elif component_type == BATCHED_DATASET:
            self.batched_dataset = load_pickle(path)
        elif component_type == MODEL_TRAIN_STATE:
            self.epoch = 0
            self.model = model.Transformer(self.config)
            self.optimizer = train.init_optimizer(self.config, self.model)
            self.losses = []

            train_state = torch.load(path)
            # load epoch + 1 so we don't retread the epoch that the model just finished
            self.epoch = train_state["epoch"] + 1
            self.model.load_state_dict(train_state["model_state"])
            self.optimizer.load_state_dict(train_state["optimizer_state"])
            self.losses = train_state["losses"]

        self.present[component_type] = True

    def clean(self, component_type):
        if not self.exists(component_type):
            return

        path = self.paths[component_type]
        Path.unlink(path)
        self.present[component_type] = False

        if component_type == RAW_DATASET:
            self.raw_dataset = None
        elif component_type == TOKENIZER:
            self.tokenizer = None
        elif component_type == UNBATCHED_DATASET:
            self.unbatched_dataset = None
        elif component_type == BATCHED_DATASET:
            self.batched_dataset = None
        elif component_type == MODEL_TRAIN_STATE:
            self.epoch = None
            self.model = None
            self.optimizer = None
            self.losses = None

    def load_all(self):
        for ct in ComponentType:
            self.load(ct)

    def clean_all(self):
        for ct in ComponentType:
            self.clean(ct)

    def create_all(self):
        self.clean_all()
        self.create(BATCHED_DATASET)
        self.create(MODEL_TRAIN_STATE)

    def status(self, component_type):
        def red(x):
            # ANSI escape characters for red
            return "\033[91m{}\033[00m".format(x)

        def green(x):
            # ANSI escape characters for green
            return "\033[92m{}\033[00m".format(x)

        cache = "cache"
        fresh = "fresh"
        colon = ": "

        def wrap(s):
            return "(" + s + ")   "

        name = component_type.name.lower()
        # length of the longest component name
        max_name_length = max([len(i.name) for i in ComponentType])
        name_str_length = max_name_length + len(colon)

        def render_absent(name, length):
            return (name + colon).ljust(length) + red("UNINITIALIZED")

        def render_present(name, length, extra):
            return (name + colon).ljust(length) + green(extra)

        if not self.present[component_type]:
            return render_absent(name, name_str_length)

        if component_type == RAW_DATASET:
            extra = f"len: {len(self.raw_dataset)}"
        elif component_type == TOKENIZER:
            extra = f"vocab: {self.tokenizer.get_vocab_size()}"
        elif component_type == UNBATCHED_DATASET:
            extra = f"len: {len(self.unbatched_dataset)}"
        elif component_type == BATCHED_DATASET:
            extra = f"batches: {len(self.batched_dataset)}"
        elif component_type == MODEL_TRAIN_STATE:
            epoch_extra = f"epoch: {self.epoch}"
            if len(self.losses) == 0:
                losses_extra = "loss: []"
            elif len(self.losses) == 1:
                losses_extra = f"loss: [{self.losses[0]}]"
            else:
                losses_extra = f"loss: [{self.losses[0]:.4f} ... {self.losses[-1]:.4f}]"

            return render_present(
                name, name_str_length, epoch_extra + ", " + losses_extra
            )

        return render_present(name, name_str_length, extra)

    def __repr__(self):
        statuses = ["  " + self.status(ct) for ct in ComponentType]
        return "Components:\n" + "\n".join(statuses)
