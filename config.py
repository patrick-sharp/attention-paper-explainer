import os
import pathlib
from pathlib import Path


# TODO: bring to parity with ToyConfig
class BaseConfig:
    num_blocks = 6
    d_model = 512
    d_ff = 2048
    num_heads = 8
    d_key = 64
    d_value = 64
    p_dropout = 0.1
    train_steps = 1e5  # also called the number of epochs
    adam_beta_1 = 0.9
    adam_beta_2 = 0.98
    adam_epsilon = 1e-9


class ToyConfig:
    # Model params
    sequence_length = 50
    num_blocks = 2  # number of blocks in decoder and encoder each
    d_model = 64
    d_ff = 256
    num_heads = 2
    d_key = 16  # size of the query and key vectors for each token
    d_value = 16  # size of the value vector for each token
    p_dropout = 0.1
    train_steps = 1e4  # also called the number of epochs
    adam_beta_1 = 0.9
    adam_beta_2 = 0.98
    adam_epsilon = 1e-9
    layer_norm_epsilon = 1e-5
    label_smoothing_epsilon = 0.1
    bias = False  # whether or not to use a bias in linear layers and LayerNorm

    # Dataset params
    num_sentence_pairs = 50000
    huggingface_cache_dir = "huggingface_cache"
    max_tokens_in_batch = 25000

    # Tokenizer params
    tokenizer_vocab_size = 16000
    tokenizer_path = "tokenizer.json"
    bos_token = "[BOS]"  # beginning of sentence token
    eos_token = "[EOS]"  # end of sentence token
    pad_token = "[PAD]"  # padding token, for padding shorter sequences to the full sequence length
    unk_token = "[UNK]"  # unknown token, for tokens that weren't in training data


class SmallConfig:
    # Model params
    sequence_length = 350  # max sentence in this config's training data is 201 tokens
    num_blocks = 2  # number of blocks in decoder and encoder each
    d_model = 256
    d_ff = 1024
    num_heads = 4
    d_key = 16  # size of the query and key vectors for each token
    d_value = 16  # size of the value vector for each token
    p_dropout = 0.1
    train_steps = 1e4  # also called the number of epochs
    adam_beta_1 = 0.9
    adam_beta_2 = 0.98
    adam_epsilon = 1e-9
    layer_norm_epsilon = 1e-5
    label_smoothing_epsilon = 0.1
    bias = False  # whether or not to use a bias in linear layers and LayerNorm

    # Dataset params
    num_sentence_pairs = 25000
    huggingface_cache_dir = "dataset_cache"

    # Tokenizer params
    tokenizer_batch_size = 1000
    tokenizer_vocab_size = 32000
    tokenizer_path = "tokenizer.json"
    unk_token = "[UNK]"  # unknown token, for tokens that weren't in training data
    bos_token = "[BOS]"  # beginning of sentence token
    eos_token = "[EOS]"  # end of sentence token
    pad_token = "[PAD]"  # padding token, for padding shorter sequences to the full sequence length


# TODO: bring to parity with ToyConfig
class BigConfig:
    num_blocks = 6
    d_model = 1024
    d_ff = 4096
    num_heads = 16
    d_key = 64
    d_value = 64
    p_dropout = 0.3
    train_steps = 3e5  # also called the number of epochs
    adam_beta_1 = 0.9
    adam_beta_2 = 0.98
    adam_epsilon = 1e-9


DEFAULT_CONFIG = ToyConfig()
