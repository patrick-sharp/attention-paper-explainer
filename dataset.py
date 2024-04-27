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


def get_raw_dataset(config):
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
        vocab_size=config.tokenizer_vocab_size,
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
    tokenizer.save(config.tokenizer_path)
    return tokenizer


def load_tokenizer(config):
    tokenizer = Tokenizer(BPE())
    return tokenizer.from_file(config.tokenizer_path)


def get_tokenizer(config, raw_dataset):
    if os.path.exists(config.tokenizer_path):
        tokenizer = load_tokenizer(config)
    else:
        if raw_dataset is None:
            raw_dataset = get_raw_dataset(config)
        tokenizer = train_tokenizer(config, raw_dataset)
    return tokenizer


def get_tokenized_dataset(config, tokenizer, raw_dataset):
    def tokenize_item(item):
        de = item["translation"]["de"]
        en = item["translation"]["en"]

        # del this key, otherwise it shows up in the returned dict
        del item["translation"]

        de_tok = tokenizer.encode(de).ids
        en_tok = tokenizer.encode(en).ids
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


class DynamicBatchedDataset(IterableDataset):
    def __init__(self, config, dataset, tokenizer):
        super().__init__()
        self.config = config
        max_tokens_in_batch = config.max_tokens_in_batch

        self.tokenizer = tokenizer
        self.dataset = dataset

        batches = []
        batch_start = 0
        max_len_de = 0
        max_len_en = 0

        print("BATCHING DATA")
        for i in tqdm(range(len(self.dataset))):
            item = self.dataset[i]
            new_max_len_de = max(max_len_de, len(item["de"]))
            new_max_len_en = max(max_len_en, len(item["en"]))

            de_exceeds = new_max_len_de * (i - batch_start + 1) > max_tokens_in_batch
            en_exceeds = new_max_len_en * (i - batch_start + 1) > max_tokens_in_batch

            if de_exceeds or en_exceeds:
                batches.append(
                    {
                        "start": batch_start,  # inclusive bound
                        "end": i,  # exclusive bound
                        "max_len_de": max_len_de,
                        "max_len_en": max_len_en,
                    }
                )
                batch_start = i
                max_len_de = len(item["de"])
                max_len_en = len(item["en"])
            else:
                max_len_de = new_max_len_de
                max_len_en = new_max_len_en
        # unless the last sentence pairs in the dataset happen to be exactly
        # the right number of tokens, we need to make them into a final
        # smaller batch
        if batches[-1]["end"] != len(self.dataset):
            batches.append(
                {
                    "start": batch_start,
                    "end": i,
                    "max_len_de": max_len_de,
                    "max_len_en": max_len_en,
                }
            )

        random.shuffle(batches)
        self.batches = batches

    def dynamic_batch_generator(self):
        pad_token = self.config.pad_token
        pad_token_id = self.tokenizer.token_to_id(pad_token)
        for batch in self.batches:
            print("PADDING SEQUENCES IN BATCH")
            batch_range = range(batch["start"], batch["end"])
            items = [self.dataset[i] for i in batch_range]
            enc_seq_len = batch["max_len_de"]
            dec_seq_len = batch["max_len_en"]
            de_padded = []
            en_padded = []

            def pad(tokens, length):
                num_pad = length - len(tokens)
                return tokens + [pad_token_id] * num_pad

            for item in items:
                de_padded.append(pad(item["de_tok"], enc_seq_len))
                en_padded.append(pad(item["en_tok"], dec_seq_len))

            # (batch_size, enc_seq_len)
            encoder_input = torch.tensor(de_padded)

            # (batch_size, dec_seq_len)
            decoder_input = torch.tensor(en_padded)

            # (batch_size, enc_seq_len)
            encoder_mask = (encoder_input != pad_token_id).int()

            # (batch_size, dec_seq_len)
            decoder_pad_mask = (decoder_input != pad_token_id).int()

            batch_size = batch["end"] - batch["start"]
            causal_mask_shape = (batch_size, dec_seq_len, dec_seq_len)

            # (batch_size, dec_seq_len, dec_seq_len)
            # only allows attention to past tokens
            # e.g.
            # [[[0., 1., 1.],
            #   [0., 0., 1.],
            #   [0., 0., 0.]]]
            decoder_causal_mask = torch.triu(
                torch.ones(causal_mask_shape), diagonal=1
            ).int()

            # (batch_size, 1, dec_seq_len)
            decoder_pad_mask = decoder_pad_mask.unsqueeze(1)

            # (batch_size, dec_seq_len, dec_seq_len)
            decoder_mask = decoder_pad_mask & decoder_causal_mask

            yield {
                "encoder_input": encoder_input,
                "decoder_input": decoder_input,
                "encoder_mask": encoder_mask,
                "decoder_mask": decoder_mask,
                # remove the eos token from the decoder input to get the labels
                "label": decoder_input[:, 1:],
                # not used by the model, just used to show progress during training
                "decoder_text": [item["de"] for item in items],
                "encoder_text": [item["en"] for item in items],
            }

    def __iter__(self):
        return self.dynamic_batch_generator()
