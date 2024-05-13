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

from torch.utils.data import Dataset


def raw_dataset(config):
    """Returns a huggingface dataset with only the first num_sentence_pairs items."""
    ds = load_dataset(
        "wmt14", "de-en", split="train", cache_dir=config.huggingface_cache_dir
    )
    ds = ds.select(range(config.num_sentence_pairs))
    return ds


def tokenizer_iterator(dataset):
    for i in range(0, 2 * len(dataset)):
        sentence_pair = dataset[i // 2]["translation"]
        if i % 2 == 0:
            yield sentence_pair["de"]
        else:
            yield sentence_pair["en"]


def init_tokenizer(config):
    tokenizer = Tokenizer(
        BPE(
            unk_token=config.unk_token,
            continuing_subword_prefix=config.csp_token,
            end_of_word_suffix=config.eow_token,
        )
    )
    return tokenizer


def train_tokenizer(config, dataset):
    tokenizer = init_tokenizer(config)
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=config.max_vocab_size,
        continuing_subword_prefix=config.csp_token,
        end_of_word_suffix=config.eow_token,
        special_tokens=[
            config.bos_token,
            config.eos_token,
            config.pad_token,
            config.unk_token,
            config.csp_token,
            config.eow_token,
        ],
    )
    tokenizer.post_processor = TemplateProcessing(
        single=f"{config.bos_token} $A {config.eos_token}",
        special_tokens=[
            (config.bos_token, 0),
            (config.eos_token, 1),
            (config.pad_token, 2),
            (config.unk_token, 3),
            (config.csp_token, 4),
            (config.eow_token, 5),
        ],
    )

    def tokenizer_iterator(dataset):
        for i in range(0, 2 * len(dataset)):
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


def create_source_mask(encoder_input, pad_token_id):
    # true for padding, false for non-padding
    # (batch_size, seq_len)
    source_mask = encoder_input == pad_token_id

    # (batch_size, 1, 1, seq_len)
    # this allows us to mask all heads of attention at once later
    source_mask = source_mask.unsqueeze(1).unsqueeze(1)
    return source_mask


def create_target_mask(decoder_input, pad_token_id):
    batch_size, seq_len = decoder_input.shape
    # (batch_size, seq_len)
    # true for padding, false for non-padding
    target_pad_mask = decoder_input == pad_token_id

    causal_mask_shape = (
        batch_size,
        1,
        seq_len,
        seq_len,
    )
    # (batch_size, 1, seq_len, seq_len
    # only allows attention to past tokens
    # e.g.
    # [[[[False., True.,  True.],
    #    [False., False., True.],
    #    [False., False., False.]]]]
    target_causal_mask = torch.triu(
        torch.ones(causal_mask_shape, dtype=torch.bool), diagonal=1
    )

    # (batch_size, 1, 1, seq_len)
    target_pad_mask = target_pad_mask.unsqueeze(1).unsqueeze(1)

    # (batch_size, 1, seq_len, seq_len)
    target_mask = target_pad_mask & target_causal_mask
    return target_mask


class BatchedDataset(Dataset):
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
        if len(batch_shapes) == 0:
            i += 1
            append_batch()
        elif batch_shapes[-1]["end"] != len(self.tokenized_dataset):
            i += 1
            append_batch()

        self.batch_shapes = batch_shapes

    def pad_batch(self, batch):
        start = batch["start"]
        end = batch["end"]
        seq_len = batch["max_len"]
        batch_size = end - start

        pad_token = self.config.pad_token
        pad_token_id = self.tokenizer.token_to_id(pad_token)
        items = [self.tokenized_dataset[i] for i in range(start, end)]
        de_padded = []
        en_padded = []

        for item in items:
            de_padded.append(pad(item["de_tok"], seq_len, pad_token_id))
            en_padded.append(pad(item["en_tok"], seq_len, pad_token_id))

        # (batch_size, seq_len)
        encoder_input = torch.tensor(de_padded, dtype=torch.int32)

        # (batch_size, seq_len)
        decoder_input = torch.tensor(en_padded, dtype=torch.int32)

        # (batch_size, 1, 1, seq_len)
        source_mask = create_source_mask(encoder_input, pad_token_id)

        # (batch_size, 1, seq_len, seq_len)
        target_mask = create_target_mask(decoder_input, pad_token_id)

        # get rid of the bos token for the labels
        # (batch_size, seq_len)
        label = decoder_input.roll(-1, 1)
        label[:, -1] = pad_token_id

        return {
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
            # not used by the model. used to show progress during training
            "encoder_text": [item["de"] for item in items],
            "decoder_text": [item["en"] for item in items],
        }

    def __len__(self):
        return len(self.batch_shapes)

    def __getitem__(self, idx):
        """Get the idx'th batch of the dataset"""
        return self.pad_batch(self.batch_shapes[idx])
