import random

import pandas as pd
import torch
from torch.utils.data import Dataset

import masking


def raw_dataset(config, split):
    if split == "train":
        filename = "toy_dataset_train.csv"
    elif split == "test":
        filename = "toy_dataset_test.csv"

    items = []
    df = pd.read_csv(filename)
    for i in range(len(df)):
        items.append(df.iloc[i].to_dict())
    return items


def tokenize_dataset(config, raw_dataset, tokenizer):
    tokenized_dataset = []
    for item in raw_dataset:
        de = item["de"]
        en = item["en"]
        de_tok = tokenizer.encode(de).ids
        en_tok = tokenizer.encode(en).ids
        assert len(de_tok) <= config.max_seq_len
        assert len(en_tok) <= config.max_seq_len
        tokenized_dataset.append(
            {
                "de": de,
                "en": en,
                "de_tok": de_tok,
                "en_tok": en_tok,
            }
        )
    return tokenized_dataset


class ToyDatasetTrain(Dataset):
    def __init__(self, components, tokenized_dataset):
        config = components.config
        de_len = max([len(item["de_tok"]) for item in tokenized_dataset])
        en_len = max([len(item["en_tok"]) for item in tokenized_dataset])

        # sample sentences to translate during model training after every epoch
        # this shows how the model is progressing
        num_examples = config.num_examples
        num_examples = min(num_examples, len(tokenized_dataset))
        example_indices = random.sample(range(len(tokenized_dataset)), num_examples)
        self.examples = []
        for i in example_indices:
            pair = tokenized_dataset[i]
            self.examples.append({"source": pair["de"], "target": pair["en"]})

        pad_token_id = components.pad_token_id

        def pad(tokens, length):
            num_pad = length - len(tokens)
            return tokens + [pad_token_id] * num_pad

        de_padded = [pad(item["de_tok"], de_len) for item in tokenized_dataset]
        en_padded = [pad(item["en_tok"], en_len) for item in tokenized_dataset]

        # (1, seq_len)
        decoder_input = torch.tensor(de_padded, dtype=torch.int32)

        # (1, seq_len)
        encoder_input = torch.tensor(en_padded, dtype=torch.int32)

        # (1, 1, 1, seq_len)
        source_mask = masking.create_source_mask(encoder_input, pad_token_id)

        # (1, 1, seq_len, seq_len)
        target_mask = masking.create_target_mask(decoder_input, pad_token_id)

        # get rid of the bos token for the labels
        # (batch_size, seq_len)
        label = decoder_input.roll(-1, 1)
        label[:, -1] = pad_token_id

        self.batch = {
            # (batch_size, de_seq_len)
            "encoder_input": encoder_input,
            # (batch_size, en_seq_len)
            "decoder_input": decoder_input,
            # (batch_size, 1, 1, de_seq_len)
            "source_mask": source_mask,
            # (batch_size, 1, seq_len, en_seq_len)
            "target_mask": target_mask,
            # (batch_size, en_seq_len-1)
            "label": label,
        }

    def __len__(self):
        if hasattr(self, "batch"):
            return 1
        else:
            return 0

    def __getitem__(self, _):
        return self.batch


class ToyDatasetTest(Dataset):
    def __init__(self, components, tokenized_dataset):
        config = components.config
        de_len = max([len(item["de_tok"]) for item in tokenized_dataset])
        en_len = max([len(item["en_tok"]) for item in tokenized_dataset])

        # sample sentences to translate during model training after every epoch
        # this shows how the model is progressing
        num_examples = config.num_examples
        num_examples = min(num_examples, len(tokenized_dataset))
        example_indices = random.sample(range(len(tokenized_dataset)), num_examples)
        self.examples = []
        for i in example_indices:
            pair = tokenized_dataset[i]
            self.examples.append({"source": pair["en"], "target": pair["de"]})

        pad_token_id = components.pad_token_id

        def pad(tokens, length):
            num_pad = length - len(tokens)
            return tokens + [pad_token_id] * num_pad

        self.batches = []
        for item in tokenized_dataset:
            de_padded = [pad(item["de_tok"], de_len)]
            en_padded = [pad(item["en_tok"], en_len)]

            # (1, seq_len)
            decoder_input = torch.tensor(de_padded, dtype=torch.int32)

            # (1, seq_len)
            encoder_input = torch.tensor(en_padded, dtype=torch.int32)

            # (1, 1, 1, seq_len)
            source_mask = masking.create_source_mask(encoder_input, pad_token_id)

            # (1, 1, seq_len, seq_len)
            target_mask = masking.create_target_mask(decoder_input, pad_token_id)

            # get rid of the bos token for the labels
            # (batch_size, seq_len)
            label = decoder_input.roll(-1, 1)
            label[:, -1] = pad_token_id

            self.batches.append(
                {
                    # (batch_size, de_seq_len)
                    "encoder_input": encoder_input,
                    # (batch_size, en_seq_len)
                    "decoder_input": decoder_input,
                    # (batch_size, 1, 1, de_seq_len)
                    "source_mask": source_mask,
                    # (batch_size, 1, seq_len, en_seq_len)
                    "target_mask": target_mask,
                    # (batch_size, en_seq_len-1)
                    "label": label,
                    "source_text": [item["en"]],
                    "target_text": [item["de"]],
                }
            )

    def __len__(self):
        if hasattr(self, "batches"):
            return len(self.batches)
        else:
            return 0

    def __getitem__(self, idx):
        return self.batches[idx]


def batch_dataset(components, tokenized_dataset, split):
    if split == "train":
        return ToyDatasetTrain(components, tokenized_dataset)
    elif split == "test":
        return ToyDatasetTest(components, tokenized_dataset)
