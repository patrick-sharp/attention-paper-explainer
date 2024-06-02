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
    target_mask = target_pad_mask | target_causal_mask
    return target_mask


class BatchedDataset(Dataset):
    config = None
    split = None  # either train or test
    tokenizer = None
    tokenized_dataset = None
    examples = None  # only for train dataset
    batch_bounds = None

    def __init__(self, components, split="train"):
        """Compute where each batch starts and ends in the dataset, and what
        sequence length each batch should have.
        Store this in self.batch_bounds"""
        super().__init__()
        self.split = split

        config = components.config
        tokenizer = components.tokenizer
        if split == "train":
            tokenized_dataset = components.train_tokenized
        elif split == "test":
            tokenized_dataset = components.test_tokenized
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
                self.examples.append({"source": pair["de"], "target": pair["en"]})

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

        # for i in tqdm(range(len(self.tokenized_dataset))):
        #    item = self.tokenized_dataset[i]
        #    new_max_len = max(max_len, len(item["de"]), len(item["en"]))

        #    if new_max_len * (i - batch_start + 1) > max_tokens_in_batch:
        #        append_batch()
        #        batch_start = i
        #        max_len = max(len(item["de"]), len(item["en"]))
        #    else:
        #        max_len = new_max_len
        ## unless the last sentence pairs in the dataset happen to be exactly
        ## the right number of tokens, we need to make them into a final
        ## smaller batch
        # if len(batch_bounds) == 0:
        #    i += 1
        #    append_batch()
        # elif batch_bounds[-1]["end"] != len(self.tokenized_dataset):
        #    i += 1
        #    append_batch()

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
            batch["source_text"] = [item["de"] for item in items]
            batch["target_text"] = [item["en"] for item in items]

        return batch

    def __len__(self):
        return len(self.batch_bounds)

    def __getitem__(self, n):
        """Get the nth batch of the dataset.
        Generated dynamically from the batch shapes, the tokenized dataset, and n."""
        return self.pad_batch(self.batch_bounds[n])
