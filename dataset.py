import random
import os

from tqdm import tqdm

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from torch.utils.data import IterableDataset


def raw_dataset(config):
    """Returns a huggingface dataset with only the first num_sentence_pairs items."""
    ds = load_dataset(
        "wmt14", "de-en", split="train", cache_dir=config.huggingface_cache_dir
    )
    ds = ds.select(range(config.num_sentence_pairs))
    return ds


def train_tokenizer(config, dataset):
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=config.vocab_size,
        special_tokens=[
            config.bos_token,
            config.eos_token,
            config.pad_token,
            config.unk_token,
        ],
    )
    tokenizer.post_processor = TemplateProcessing(
        single=f"{config.bos_token} $A {config.eos_token}",
        special_tokens=[
            (config.bos_token, 0),
            (config.eos_token, 1),
            (config.pad_token, 2),
            (config.unk_token, 3),
        ],
    )

    def tokenizer_iterator(dataset):
        for i in range(0, len(dataset)):
            sentence_pair = dataset[i // 2]["translation"]
            if i % 2 == 0:
                yield sentence_pair["de"]
            else:
                yield sentence_pair["en"]

    tokenizer.train_from_iterator(
        tokenizer_iterator(dataset),
        trainer=trainer,
    )
    return tokenizer


def tokenize_dataset(config, raw_dataset, tokenizer):
    def tokenize_item(item):
        de = item["translation"]["de"]
        en = item["translation"]["en"]

        # del this key, otherwise it shows up in the returned dict
        del item["translation"]

        de_tok = tokenizer.encode(de).ids
        en_tok = tokenizer.encode(en).ids
        assert len(de_tok) <= config.max_sequence_length
        assert len(en_tok) <= config.max_sequence_length
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


class BatchedDataset(IterableDataset):
    def __init__(self, config, tokenizer, tokenized_dataset):
        """Compute where each batch starts and ends in the dataset, and what
        sequence length each batch should have.
        Store this in self.batch_shapes"""
        super().__init__()
        self.config = config
        max_tokens_in_batch = config.max_tokens_in_batch

        self.tokenizer = tokenizer
        self.tokenized_dataset = tokenized_dataset

        batch_shapes = []
        batch_start = 0
        max_len = 0  # this is the maximum length of either de or en

        print("Batching data...")
        i = 0

        def append_batch():
            batch_shapes.append(
                {
                    "start": batch_start,  # inclusive bound
                    "end": i,  # exclusive bound
                    "max_len": max_len,
                }
            )

        for i in tqdm(range(len(self.tokenized_dataset))):
            item = self.tokenized_dataset[i]
            new_max_len = max(max_len, len(item["de"]), len(item["en"]))

            if new_max_len * (i - batch_start + 1) > max_tokens_in_batch:
                append_batch()
                batch_start = i
                max_len = max(len(item["de"]), len(item["en"]))
            else:
                max_len = new_max_len
        # unless the last sentence pairs in the dataset happen to be exactly
        # the right number of tokens, we need to make them into a final
        # smaller batch
        if batch_shapes[-1]["end"] != len(self.tokenized_dataset):
            append_batch()

        random.shuffle(batch_shapes)
        self.batch_shapes = batch_shapes

    def pad_batch(self, batch):
        start = batch["start"]
        end = batch["end"]
        sequence_length = batch["max_len"]
        batch_size = end - start

        pad_token = self.config.pad_token
        pad_token_id = self.tokenizer.token_to_id(pad_token)
        items = [self.tokenized_dataset[i] for i in range(start, end)]
        de_padded = []
        en_padded = []

        def pad(tokens, length):
            num_pad = length - len(tokens)
            return tokens + [pad_token_id] * num_pad

        for item in items:
            de_padded.append(pad(item["de_tok"], sequence_length))
            en_padded.append(pad(item["en_tok"], sequence_length))

        # (batch_size, sequence_length)
        encoder_input = torch.tensor(de_padded)

        # (batch_size, sequence_length)
        decoder_input = torch.tensor(en_padded)

        # true for padding, false for non-padding
        # (batch_size, sequence_length)
        source_mask = encoder_input == pad_token_id

        # (batch_size, 1, 1, sequence_length)
        # this allows us to mask all heads of attention at once later
        source_mask = source_mask.unsqueeze(1).unsqueeze(1)

        # (batch_size, sequence_length)
        # true for padding, false for non-padding
        target_pad_mask = decoder_input == pad_token_id

        causal_mask_shape = (
            batch_size,
            1,
            sequence_length,
            sequence_length,
        )

        # (batch_size, 1, sequence_length, sequence_length
        # only allows attention to past tokens
        # e.g.
        # [[[[False., True.,  True.],
        #    [False., False., True.],
        #    [False., False., False.]]]]
        target_causal_mask = torch.triu(
            torch.ones(causal_mask_shape, dtype=torch.bool), diagonal=1
        )

        # (batch_size, 1, 1, sequence_length)
        target_pad_mask = target_pad_mask.unsqueeze(1).unsqueeze(1)

        # (batch_size, 1, sequence_length, sequence_length)
        target_mask = target_pad_mask & target_causal_mask

        # get rid of the bos token for the labels
        # (batch_size, sequence_length)
        label = decoder_input.roll(-1, 1)
        label[:, -1] = pad_token_id

        return {
            # (batch_size, sequence_length)
            "encoder_input": encoder_input,
            # (batch_size, sequence_length)
            "decoder_input": decoder_input,
            # (batch_size, 1, 1, sequence_length)
            "source_mask": source_mask,
            # (batch_size, 1, sequence_length, sequence_length)
            "target_mask": target_mask,
            # (batch_size, sequence_length)
            "label": label,
            # not used by the model. used to show progress during training
            "decoder_text": [item["de"] for item in items],
            "encoder_text": [item["en"] for item in items],
        }

    def __len__(self):
        return len(self.batch_shapes)

    def __getitem__(self, idx):
        """Get the idx'th batch of the dataset"""
        return self.pad_batch(self.batch_shapes[idx])

    def dynamic_batch_generator(self):
        """Generator for implementing the iterable protocol"""
        for batch in self.batch_shapes:
            yield self.pad_batch(batch)

    def __iter__(self):
        return self.dynamic_batch_generator()
