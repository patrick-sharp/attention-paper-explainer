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
    TRAIN_RAW = 0
    TOKENIZER = 1
    TRAIN_TOKENIZED = 2
    TRAIN_BATCHED = 3
    MODEL_TRAIN_STATE = 4
    TEST_RAW = 5
    TEST_TOKENIZED = 6
    TEST_BATCHED = 7


dependencies = {
    TRAIN_RAW: [],
    TOKENIZER: [TRAIN_RAW],
    TRAIN_TOKENIZED: [TRAIN_RAW, TOKENIZER],
    TRAIN_BATCHED: [TRAIN_TOKENIZED],
    MODEL_TRAIN_STATE: [],
    TEST_RAW: [],
    TEST_TOKENIZED: [TEST_RAW, TOKENIZER],
    TEST_BATCHED: [TEST_TOKENIZED],
}


class Components:
    config = None

    train_raw = None
    tokenizer = None
    train_tokenized = None
    train_batched = None

    epoch = None
    model = None
    optimizer = None
    losses = None
    translations = None

    test_raw = None
    test_tokenized = None
    test_batched = None

    def __init__(self, config):
        folder = os.path.join(config.components_folder, config.name)
        Path(folder).mkdir(parents=True, exist_ok=True)

        self.config = config

        if config.use_random_seed:
            self.set_seeds()

        # allows us to access component types with reference equality in the train loop
        # and in single-sentence translation
        self.types = ComponentType

        self.paths = {}
        for component_type in self.types:
            filename = getattr(config, component_type.name.lower() + "_filename")
            path = os.path.join(folder, filename)
            self.paths[component_type] = path

        self.present = {component_type: False for component_type in ComponentType}

        self.load_all()

    def set_seeds(self):
        """Keep in mind that the torch generator doesn't reset unless you exit and restart the repl."""
        seed = self.config.random_seed
        random.seed(seed)
        torch.manual_seed(seed)

    def exists(self, component_type):
        """return whether a cached version of this component exists"""
        path = self.paths[component_type]
        return os.path.exists(path)

    def create(self, component_type):
        """create and save a component"""
        self.clean(component_type)

        for dependency in dependencies[component_type]:
            if not self.present[dependency]:
                self.create(dependency)

        name = component_type.name.lower().replace("_", " ")

        print("Initializing " + name + "...")
        if component_type.value == TRAIN_RAW.value:
            component = dataset.raw_dataset(self.config, split="train")
            self.train_raw = component
        elif component_type == TOKENIZER:
            component = dataset.train_tokenizer(self.config, self.train_raw)
            self.tokenizer = component
        elif component_type == TRAIN_TOKENIZED:
            component = dataset.tokenize_dataset(
                self.config, self.train_raw, self.tokenizer
            )
            self.train_tokenized = component
        elif component_type == TRAIN_BATCHED:
            component = dataset.BatchedDataset(self)
            self.train_batched = component
        elif component_type == MODEL_TRAIN_STATE:
            # model train state is a special case.
            # it doesn't depend on other components since we want to be able to use it
            # to translate an arbitrary sentence without loading the dataset
            self.clean(MODEL_TRAIN_STATE)
            self.fresh_train_state()
            # don't set component; the train loop handles saving the state
        elif component_type == TEST_RAW:
            component = dataset.raw_dataset(self.config, split="test")
            self.test_raw = component
        elif component_type == TEST_TOKENIZED:
            component = dataset.tokenize_dataset(
                self.config, self.test_raw, self.tokenizer
            )
            self.test_tokenized = component
        elif component_type == TEST_BATCHED:
            component = dataset.BatchedDataset(self, split="test")
            self.test_batched = component

        # the train loop handles initializing and saving the model state
        if component_type != MODEL_TRAIN_STATE:
            self.save(component_type, component)

        self.present[component_type] = True

    def save(self, component_type, component):
        path = self.paths[component_type]
        if component_type == TRAIN_RAW:
            save_pickle(component, path)
        elif component_type == TOKENIZER:
            self.tokenizer.save(path)
        elif component_type == TRAIN_TOKENIZED:
            save_pickle(component, path)
        elif component_type == TRAIN_BATCHED:
            save_pickle(component, path)
        elif component_type == MODEL_TRAIN_STATE:
            torch.save(component, path)
        elif component_type == TEST_RAW:
            save_pickle(component, path)
        elif component_type == TEST_TOKENIZED:
            save_pickle(component, path)
        elif component_type == TEST_BATCHED:
            save_pickle(component, path)

    def load(self, component_type):
        if not self.exists(component_type):
            return

        path = self.paths[component_type]

        try:
            if component_type == TRAIN_RAW:
                self.train_raw = load_pickle(path)
            elif component_type == TOKENIZER:
                tokenizer = dataset.init_tokenizer(self.config)
                tokenizer = tokenizer.from_file(path)
                self.tokenizer = tokenizer
            elif component_type == TRAIN_TOKENIZED:
                self.train_tokenized = load_pickle(path)
            elif component_type == TRAIN_BATCHED:
                self.train_batched = load_pickle(path)
            elif component_type == MODEL_TRAIN_STATE:
                self.fresh_train_state()
                train_state = torch.load(path)
                # load epoch + 1 so we don't retread the epoch that the model just finished
                self.epoch = train_state["epoch"] + 1
                self.model.load_state_dict(train_state["model_state"])
                self.optimizer.load_state_dict(train_state["optimizer_state"])
                self.losses = train_state["losses"]
                self.translations = train_state["translations"]
            elif component_type == TEST_RAW:
                self.test_raw = load_pickle(path)
            elif component_type == TEST_TOKENIZED:
                self.test_tokenized = load_pickle(path)
            elif component_type == TEST_BATCHED:
                self.test_batched = load_pickle(path)

            self.present[component_type] = True
        except Exception as ex:
            # If the state on disk is bad, delete it
            print(ex)
            print(
                f"Exception loading {component_type.name}, deleting cached version..."
            )
            self.clean(component_type)

    def clean(self, component_type):
        """cleans a component and any components that depend on it"""

        # if any components depend on this one, clean them too
        for component_type_other in self.types:
            if component_type in dependencies[component_type_other]:
                self.clean(component_type_other)

        path = self.paths[component_type]
        # deletes the file for this component
        Path.unlink(path, missing_ok=True)
        self.present[component_type] = False

        if component_type == TRAIN_RAW:
            self.train_raw = None
        elif component_type == TOKENIZER:
            self.tokenizer = None
        elif component_type == TRAIN_TOKENIZED:
            self.train_tokenized = None
        elif component_type == TRAIN_BATCHED:
            self.train_batched = None
        elif component_type == MODEL_TRAIN_STATE:
            self.epoch = None
            self.model = None
            self.optimizer = None
            self.losses = None
            self.translations = None
        elif component_type == TEST_RAW:
            self.test_raw = None
        elif component_type == TEST_TOKENIZED:
            self.test_tokenized = None
        elif component_type == TEST_BATCHED:
            self.test_batched = None

    def fresh_train_state(self):
        self.epoch = 0
        self.model = model.Transformer(self)
        self.optimizer = train.init_optimizer(self.config, self.model)
        self.losses = []
        self.translations = []

    def load_all(self):
        for ct in ComponentType:
            self.load(ct)

    def clean_all(self):
        for ct in ComponentType:
            self.clean(ct)

    def init_all(self):
        """Initialize all components, preferring to retrieve from cache"""
        for type in self.types:
            if not self.present[type]:
                if self.exists(type):
                    self.load(type)
                else:
                    self.create(type)

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

        if component_type == TRAIN_RAW:
            extra = f"len: {len(self.train_raw)}"
        elif component_type == TOKENIZER:
            extra = f"vocab: {self.tokenizer.get_vocab_size()}"
        elif component_type == TRAIN_TOKENIZED:
            extra = f"len: {len(self.train_tokenized)}"
        elif component_type == TRAIN_BATCHED:
            extra = f"batches: {len(self.train_batched)}"
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
        elif component_type == TEST_RAW:
            extra = f"len: {len(self.test_raw)}"
        elif component_type == TEST_TOKENIZED:
            extra = f"len: {len(self.test_tokenized)}"
        elif component_type == TEST_BATCHED:
            extra = f"batches: {len(self.test_batched)}"

        return render_present(name, name_str_length, extra)

    def __repr__(self):
        statuses = ["  " + self.status(ct) for ct in ComponentType]
        return "Components:\n" + "\n".join(statuses)
