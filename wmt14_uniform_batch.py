import random

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
    # get rid of the middleman "translation" key in each sentence pair
    ds = ds.map(itemgetter("translation"), remove_columns=["translation"])
    return ds


def tokenize_dataset(config, raw_dataset, tokenizer):
    def tokenize_item(item):
        en = item["en"]
        de = item["de"]

        en_tok = tokenizer.encode(en).ids
        de_tok = tokenizer.encode(de).ids
        assert len(en_tok) <= config.max_seq_len
        assert len(de_tok) <= config.max_seq_len
        return {
            "en_tok": en_tok,
            "de_tok": de_tok,
        }

    ds = raw_dataset.map(tokenize_item)
    return ds


def batch_dataset(components, tokenized_dataset, split):
    config = components.config
    pad_token = config.pad_token
    pad_token_id = components.tokenizer.token_to_id(pad_token)
    max_language_tokens_in_batch = config.max_language_tokens_in_batch

    max_len = 0

    def update_max_len(item):
        nonlocal max_len
        max_len = max(len(item["en_tok"]), len(item["de_tok"]), max_len)

    tokenized_dataset.map(update_max_len, desc="Calculating max length")

    sentences_per_batch = max_language_tokens_in_batch // max_len

    def pad_sentence(token_ids, length, pad_token_id):
        num_pad = length - len(token_ids)
        return token_ids + [pad_token_id] * num_pad

    def pad_batch(batch):
        en_padded = []
        de_padded = []

        for token_ids in batch["en_tok"]:
            en_padded.append(pad_sentence(token_ids, max_len, pad_token_id))

        for token_ids in batch["de_tok"]:
            de_padded.append(pad_sentence(token_ids, max_len, pad_token_id))

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

        # values of this dict are all lists of length one to turn the whole batch into
        # one dataset entry. we do this so that we can use batch_size=None during
        # training, which comes in handy when we use a more complex batching scheme
        padded_batch = {
            # (batch_size, seq_len)
            "encoder_input": [encoder_input],
            # (batch_size, seq_len)
            "decoder_input": [decoder_input],
            # (batch_size, 1, 1, seq_len)
            "source_mask": [source_mask],
            # (batch_size, 1, seq_len, seq_len)
            "target_mask": [target_mask],
            # (batch_size, seq_len)
            "label": [label],
        }

        # we need target text for computing BLEU
        # source text is useful to have for debugging / instrumentation
        if split == "test":
            padded_batch["source_text"] = [[item for item in batch["en"]]]
            padded_batch["target_text"] = [[item for item in batch["de"]]]

        return padded_batch

    batched_dataset = tokenized_dataset.map(
        pad_batch,
        batched=True,
        batch_size=sentences_per_batch,
        remove_columns=["en", "de", "en_tok", "de_tok"],
        desc="Batching",
    )

    # otherwise it will convert the tensors we created in pad_batch into lists
    batched_dataset.set_format("torch")

    return batched_dataset
