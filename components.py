import os
import pickle
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE

from component_enum import *
from printing import red, green, print_clean_exception_traceback
import bpe_tokenizer
import model
import training


def save_pickle(obj, path):
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


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
        self.device = torch.device(config.device_string)

        self.paths = {}
        for component_type in ComponentType:
            filename = getattr(config, component_type.name.lower() + "_filename")
            path = os.path.join(folder, filename)
            self.paths[component_type] = path

        self.present = {component_type: False for component_type in ComponentType}

        self.load_all()

    def cached(self, component_type):
        """return whether a cached version of this component exists"""
        path = self.paths[component_type]
        return os.path.exists(path)

    def create(self, component_type):
        """create and save a component"""
        self.clean(component_type)
        config = self.config
        dataset_module = config.dataset_module

        for dependency in component_dependencies[component_type]:
            if not self.present[dependency]:
                self.create(dependency)

        name = component_type.name.lower().replace("_", " ")

        print("Initializing " + name + "...")
        if component_type == TRAIN_RAW:
            component = dataset_module.raw_dataset(config, split="train")
            self.train_raw = component
        elif component_type == TOKENIZER:
            component = bpe_tokenizer.train_tokenizer(config, self.train_raw)
            self.tokenizer = component
            self.set_special_token_ids()
        elif component_type == TRAIN_TOKENIZED:
            component = dataset_module.tokenize_dataset(
                config, self.train_raw, self.tokenizer
            )
            self.train_tokenized = component
        elif component_type == TRAIN_BATCHED:
            component = dataset_module.batch_dataset(
                self, self.train_tokenized, split="train"
            )
            self.train_batched = component
        elif component_type == MODEL_TRAIN_STATE:
            # model train state is a special case.
            # it doesn't depend on other components since we want to be able to use it
            # to translate an arbitrary sentence without loading the dataset
            self.clean(MODEL_TRAIN_STATE)
            self.fresh_train_state()
            # don't set component; the train loop handles saving the state
        elif component_type == TEST_RAW:
            component = dataset_module.raw_dataset(config, split="test")
            self.test_raw = component
        elif component_type == TEST_TOKENIZED:
            component = dataset_module.tokenize_dataset(
                self.config, self.test_raw, self.tokenizer
            )
            self.test_tokenized = component
        elif component_type == TEST_BATCHED:
            component = dataset_module.batch_dataset(
                self, self.test_tokenized, split="test"
            )
            self.test_batched = component

        # the train loop handles initializing and saving the model state
        if component_type != MODEL_TRAIN_STATE:
            self.save(component_type, component)

        self.present[component_type] = True
        print("Done initializing " + name)

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
        if not self.cached(component_type):
            return

        path = self.paths[component_type]

        try:
            if component_type == TRAIN_RAW:
                self.train_raw = load_pickle(path)
            elif component_type == TOKENIZER:
                tokenizer = bpe_tokenizer.init_tokenizer(self.config)
                tokenizer = tokenizer.from_file(path)
                self.tokenizer = tokenizer
                self.set_special_token_ids()
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
            msg = f"Exception loading {component_type.name}, deleting cached version..."
            print(red(msg))
            print_clean_exception_traceback(ex)
            print()
            self.clean(component_type)

    def require(self, component_type):
        """If a component is not initialized, initialize it. Prefer to initialize from cache"""
        if not self.present[component_type]:
            if self.cached(component_type):
                self.load(component_type)
            else:
                self.create(component_type)

    def clean(self, component_type):
        """cleans a component and any components that depend on it"""

        # if any components depend on this one, clean them too
        for component_type_other in ComponentType:
            if component_type in component_dependencies[component_type_other]:
                self.clean(component_type_other)

        path = self.paths[component_type]
        # deletes the file for this component
        Path.unlink(path, missing_ok=True)
        self.present[component_type] = False

        if component_type == TRAIN_RAW:
            self.train_raw = None
        elif component_type == TOKENIZER:
            self.tokenizer = None
            self.bos_token_id = None
            self.eos_token_id = None
            self.pad_token_id = None
            self.csp_token_id = None
            self.eow_token_id = None
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
        self.optimizer = training.init_optimizer(self.config, self.model)
        self.losses = []
        self.translations = []

    def set_special_token_ids(self):
        self.bos_token_id = self.tokenizer.token_to_id(self.config.bos_token)
        self.eos_token_id = self.tokenizer.token_to_id(self.config.eos_token)
        self.pad_token_id = self.tokenizer.token_to_id(self.config.pad_token)
        self.csp_token_id = self.tokenizer.token_to_id(self.config.csp_token)
        self.eow_token_id = self.tokenizer.token_to_id(self.config.eow_token)

    def create_all(self):
        for type in ComponentType:
            self.create(type)

    def load_all(self):
        for ct in ComponentType:
            self.load(ct)

    def require_all(self):
        """Initialize all components, preferring to retrieve from cache"""
        for ct in ComponentType:
            self.require(ct)

    def clean_all(self):
        for ct in ComponentType:
            self.clean(ct)

    def component_status(self, component_type):
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
        colon = ": "
        max_name_length = max([len(i.name) for i in ComponentType])
        name_str_length = max_name_length + len(colon)

        config_status = (
            "  " + "config: ".ljust(name_str_length) + self.config.name + "\n"
        )
        device_status = (
            "  " + "device: ".ljust(name_str_length) + self.config.device_string + "\n"
        )
        statuses = ["  " + self.component_status(ct) for ct in ComponentType]
        return "Components:\n" + config_status + device_status + "\n".join(statuses)
