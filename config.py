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
    random_seed = 0  # used for python standard lib and for pytorch
    components_folder = "components"  # stores cached components

    # Model params
    max_sequence_length = 1500  # only used for translation, not training
    num_blocks = 2  # number of blocks in decoder and encoder each
    d_model = 64
    d_ff = 256
    num_heads = 2
    d_key = 16  # size of the query and key vectors for each token
    d_value = 16  # size of the value vector for each token
    p_dropout = 0.1
    bias = False  # whether or not to use a bias in linear layers and LayerNorm

    # model training params
    train_steps = 10_000  # also called the number of epochs
    adam_learning_rate = 1e-4
    adam_beta_1 = 0.9
    adam_beta_2 = 0.98
    adam_epsilon = 1e-9
    layer_norm_epsilon = 1e-5
    label_smoothing_epsilon = 0.1
    train_state_filename = "train_state.pt"

    # Dataset params
    num_sentence_pairs = 10000
    huggingface_cache_dir = "huggingface_cache"
    raw_dataset_filename = "raw_dataset.pkl"
    unbatched_dataset_filename = "unbatched_dataset.pkl"
    batched_dataset_filename = "batched_dataset.pkl"
    max_tokens_in_batch = 10000

    # Tokenizer params
    vocab_size = 8000
    tokenizer_filename = "tokenizer.json"
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
