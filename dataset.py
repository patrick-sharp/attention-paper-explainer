from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.data import IterableDataset

from config import DEFAULT_CONFIG


def get_dataset(config=DEFAULT_CONFIG):
    """Returns a huggingface dataset with only the first num_sentence_pairs items"""
    full_dataset = load_dataset(
        "wmt14", "de-en", split="train", cache_dir=config.huggingface_cache_dir
    )
    return full_dataset.select(range(config.num_sentence_pairs))


def dataset_iterator(dataset):
    for i in range(0, len(dataset)):
        sentence_pair = dataset[i // 2]["translation"]
        if i % 2 == 0:
            yield sentence_pair["de"]
        else:
            yield sentence_pair["en"]


def train_tokenizer(config=DEFAULT_CONFIG, dataset=None):
    if dataset is None:
        dataset = get_dataset(config)
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

    tokenizer.train_from_iterator(
        dataset_iterator(dataset),
        trainer=trainer,
    )
    tokenizer.save(config.tokenizer_path)
    return tokenizer, dataset


def load_tokenizer(config=DEFAULT_CONFIG):
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer.from_file(config.tokenizer_path)


def get_tokenizer(config=DEFAULT_CONFIG):
    if os.path.exists(config.tokenizer_path):
        tokenizer = dataset.load_tokenizer(config)
    else:
        tokenizer = dataset.train_tokenizer(config)
    return tokenizer


def DynamicBatchedDataset(IterableDataset):
    def __init__(self, config):
        super()
        max_tokens_in_batch = config.max_tokens_in_batch

        self.config = config
        self.dataset = get_dataset(config)
        self.tokenizer = get_tokenizer(config, self.dataset)

        tokenized = [None] * len(ds)
        for i, item in tqdm(enumerate(ds)):
            sentence_pair = item["translation"]
            tokenized[i] = {
                "de": tokenizer.encode(sentence_pair["de"]).ids,
                "en": tokenizer.encode(sentence_pair["en"]).ids,
            }

        def token_sum(tokenized_pair):
            de_tokens = len(tokenized_pair["de"])
            en_tokens = len(tokenized_pair["en"])
            return de_tokens + en_tokens

        tokenized.sort(key=token_sum)

        batches = []
        batch_total_tokens = 0
        #for tokenized_pair in tokenized:


    def __iter__(self):
        pass
