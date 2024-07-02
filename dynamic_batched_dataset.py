import random
import os
import copy

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from operator import itemgetter
import torch
from tqdm import tqdm

from torch.utils.data import Dataset

import masking


def raw_dataset(config, split):
    """Returns the wmt14 huggingface dataset."""
    ds = load_dataset(
        "wmt14", "de-en", split=split, cache_dir=config.huggingface_cache_dir
    )
    # only return the dataset with a subset of the rows
    if split == "train":
        num_pairs = config.train_sentence_pairs
    elif split == "test":
        num_pairs = config.test_sentence_pairs
    ds = ds.select(range(num_pairs))
    ds = ds.map(itemgetter("translation"))
    return ds


def tokenize_dataset(config, raw_dataset, tokenizer):
    def tokenize_item(item):
        de = item["translation"]["de"]
        en = item["translation"]["en"]

        # delete this key, otherwise it shows up in the returned dict
        del item["translation"]

        de_tok = tokenizer.encode(de).ids
        en_tok = tokenizer.encode(en).ids
        assert len(de_tok) <= config.max_seq_len
        assert len(en_tok) <= config.max_seq_len
        return {
            "de": de,
            "en": en,
            "de_tok": de_tok,
            "en_tok": en_tok,
            "length": len(de_tok) + len(en_tok),
        }

    ds = raw_dataset.map(tokenize_item)
    ds = ds.sort("length", reverse=True)
    return ds


def pad(tokens, length, pad_token_id):
    num_pad = length - len(tokens)
    return tokens + [pad_token_id] * num_pad


class BatchedDataset(Dataset):
    config = None
    split = None  # either train or test
    tokenizer = None
    tokenized_dataset = None
    examples = None  # only for train dataset
    batch_bounds = None

    def __init__(self, components, tokenized_dataset, split="train"):
        """Compute where each batch starts and ends in the dataset, and what
        sequence length each batch should have.
        Store this in self.batch_bounds"""
        super().__init__()
        self.split = split

        config = components.config
        tokenizer = components.tokenizer
        self.config = config
        max_tokens_in_batch = config.max_tokens_in_batch

        self.tokenizer = tokenizer
        self.tokenized_dataset = tokenized_dataset

        # sample sentences to translate during model training after every epoch
        # this shows how the model is progressing
        if split == "train":
            num_examples = config.num_examples
            num_examples = min(num_examples, len(tokenized_dataset))
            example_indices = random.sample(range(len(tokenized_dataset)), num_examples)
            self.examples = []
            for i in example_indices:
                pair = tokenized_dataset[i]
                self.examples.append({"source": pair["en"], "target": pair["de"]})

        batch_bounds = []

        def append_batch(start, end, max_len, md=0, me=0):
            batch_bounds.append(
                {
                    "start": start,  # inclusive bound
                    "end": end,  # exclusive bound
                    "max_len": max_len,  # length of longest sequence in batch
                    "md": md,
                    "me": me,
                }
            )

        batch_start = 0
        max_len = 0  # this is the maximum length of either de or en
        md = 0
        me = 0
        for i in tqdm(range(len(tokenized_dataset))):
            curr = self.tokenized_dataset[i]
            # max len if we add curr to the batch
            with_curr_max_len = max(max_len, len(curr["de"]), len(curr["en"]))
            with_curr_md = max(md, len(curr["de"]))
            with_curr_me = max(me, len(curr["en"]))

            pairs_in_batch = i - batch_start + 1
            with_curr_tokens = with_curr_max_len * pairs_in_batch * 2

            if with_curr_tokens > max_tokens_in_batch:
                # curr is in the next batch
                append_batch(
                    batch_start, i, with_curr_max_len, with_curr_md, with_curr_me
                )
                batch_start = i
                max_len = max(len(curr["de"]), len(curr["en"]))
                md = len(curr["de"])
                me = len(curr["en"])
            else:
                # curr is in this batch
                max_len = with_curr_max_len
                md = with_curr_md
                me = with_curr_me

        # we will always have at least one trailing element, so we need to append
        # one final batch
        append_batch(batch_start, len(tokenized_dataset), max_len, md, me)

        self.batch_bounds = batch_bounds

    def pad_batch(self, batch):
        start = batch["start"]
        end = batch["end"]
        seq_len = batch["max_len"]
        batch_size = end - start
        split = self.tokenized_dataset.split

        pad_token = self.config.pad_token
        pad_token_id = self.tokenizer.token_to_id(pad_token)
        items = [self.tokenized_dataset[i] for i in range(start, end)]
        de_padded = []
        en_padded = []

        for item in items:
            de_padded.append(pad(item["de_tok"], seq_len, pad_token_id))
            en_padded.append(pad(item["en_tok"], seq_len, pad_token_id))

        # (batch_size, seq_len)
        encoder_input = torch.tensor(en_padded, dtype=torch.int32)

        # (batch_size, seq_len)
        decoder_input = torch.tensor(de_padded, dtype=torch.int32)

        # (batch_size, 1, 1, seq_len)
        source_mask = masking.create_source_mask(encoder_input, pad_token_id)

        # (batch_size, 1, seq_len, seq_len)
        target_mask = masking.create_target_mask(decoder_input, pad_token_id)

        # get rid of the bos token for the labels
        # (batch_size, seq_len)
        label = decoder_input.roll(-1, 1)
        label[:, -1] = pad_token_id

        batch = {
            # (batch_size, seq_len)
            "encoder_input": encoder_input,
            # (batch_size, seq_len)
            "decoder_input": decoder_input,
            # (batch_size, 1, 1, seq_len)
            "source_mask": source_mask,
            # (batch_size, 1, seq_len, seq_len)
            "target_mask": target_mask,
            # (batch_size, seq_len)
            "label": label,
        }

        # we need target text for computing BLEU
        # source text is useful to have for debugging / instrumentation
        if split == "test":
            batch["source_text"] = [item["en"] for item in items]
            batch["target_text"] = [item["de"] for item in items]

        return batch

    def __len__(self):
        return len(self.batch_bounds)

    def __getitem__(self, n):
        """Get the nth batch of the dataset.
        Generated dynamically from the batch shapes, the tokenized dataset, and n."""
        return self.pad_batch(self.batch_bounds[n])


class SyntheticDataset(Dataset):
    def __init__(self, filename, split):
        self.split = split
        self.items = []
        df = pd.read_csv(filename)
        for i in range(len(df)):
            self.items.append({"translation": df.iloc[i].to_dict()})

    def __len__(self):
        if hasattr(self, "items"):
            return len(self.items)
        else:
            return 0

    def __getitem__(self, idx):
        return self.items[idx]

    def sort(self, key, reverse=False):
        def keyfunc(x):
            return x[key]

        self.items.sort(key=keyfunc, reverse=reverse)
        return self

    def map(self, func):
        new_ds = copy.deepcopy(self)
        for i, item in enumerate(new_ds.items):
            new_ds.items[i] = func(item)
        return new_ds


def batch_dataset(components, tokenized_dataset, split):
    return BatchedDataset(components, tokenized_dataset, split)
