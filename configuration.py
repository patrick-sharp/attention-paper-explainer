import os
import pathlib
from pathlib import Path

import torch

import toy_dataset
import dynamic_batched_dataset

# this gets rid of an annoying huggingface error message about deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BaseConfig:
    """The base parameter values from the paper"""

    name = "BaseConfig"

    components_folder = "components"  # stores cached components

    # Model params
    device_string = "cpu"  # what pytorch backend to use (cpu, cuda, mkl, mps, etc.)
    max_seq_len = 1500  # maximum length of input sequence the model will accept
    max_translation_len = max_seq_len  # maximum length of output translation
    num_blocks = 6  # number of blocks in decoder and encoder each
    d_model = 512
    d_ff = 2048
    num_heads = 8
    d_key = 64  # size of the query and key vectors for each token
    d_value = 64  # size of the value vector for each token
    p_dropout = 0.1
    bias = False  # whether or not to use a bias in linear layers and LayerNorm
    flash_attention = False  # whether or not to use the flash attention cuda kernel

    # Dataset params
    dataset_module = toy_dataset
    num_examples = 5  # number of example pairs the model will keep track of during training to give an idea of how well it's doing
    train_raw_filename = "train_raw.pkl"
    train_tokenized_filename = "train_tokenized.pkl"
    train_batched_filename = "train_batched.pkl"
    test_raw_filename = "test_raw.pkl"
    test_tokenized_filename = "test_tokenized.pkl"
    test_batched_filename = "test_batched.pkl"

    # wmt14 dataset params
    huggingface_cache_dir = "huggingface_cache"
    max_tokens_in_batch = 10000
    train_sentence_pairs = 10000
    test_sentence_pairs = 1000

    # Tokenizer params
    max_vocab_size = 8000  # actual size may be less if the data is small
    tokenizer_filename = "tokenizer.json"
    bos_token = "[BOS]"  # beginning of sentence token
    eos_token = "[EOS]"  # end of sentence token
    pad_token = "[PAD]"  # padding token, for padding shorter sequences to the full sequence length
    unk_token = "[UNK]"  # unknown token, for tokens that weren't in training data
    csp_token = "[CSP]"  # continuing subword prefix, indicates that this token is not the start of a word
    eow_token = "[EOW]"  # end of word token

    # Model training params
    train_steps = 50  # also called the number of epochs
    adam_learning_rate = 1e-4
    adam_beta_1 = 0.9
    adam_beta_2 = 0.98
    adam_epsilon = 1e-9
    layer_norm_epsilon = 1e-5
    label_smoothing_epsilon = 0.1
    model_train_state_filename = "model_train_state.pt"

    # Translation params
    beam_width = 4
    length_penalty_alpha = 0.6


class ToyConfig(BaseConfig):
    """Very small parameter values"""

    name = "ToyConfig"

    dataset_module = toy_dataset
    max_translation_len = 50

    d_model = 64
    d_ff = 256
    num_heads = 4
    d_key = 16
    d_value = 16
    num_blocks = 1

    train_steps = 115
    adam_learning_rate = 1e-3


class ToyDynamicBatchedConfig(BaseConfig):
    """Very small parameter values"""

    name = "ToyDynamicBatchedConfig"

    dataset_module = dynamic_batched_dataset
    max_translation_len = 50

    d_model = 64
    d_ff = 256
    num_heads = 4
    d_key = 16
    d_value = 16
    num_blocks = 1

    train_steps = 115
    adam_learning_rate = 1e-3


class SmallConfig(BaseConfig):
    """Smaller than BaseConfig, but still uses the wmt14 data"""

    name = "SmallConfig"

    d_model = 256
    d_ff = 512
    num_heads = 4
    num_blocks = 2

    max_translation_len = 50
    train_steps = 100
    train_sentence_pairs = 1
    test_sentence_pairs = 1


class TwoSentenceConfig(BaseConfig):
    name = "TwoSentenceConfig"
    max_translation_len = 20
    train_steps = 30
    train_sentence_pairs = 2
    test_sentence_pairs = 2


# The big model params from the paper
class BigConfig(BaseConfig):
    name = "BigConfig"


CONFIG = ToyDynamicBatchedConfig()
