from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


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
            sentence_pair = dataset[i // 2]
            if i % 2 == 0:
                yield sentence_pair["de"]
            else:
                yield sentence_pair["en"]

    tokenizer.train_from_iterator(
        tokenizer_iterator(dataset),
        trainer=trainer,
    )
    return tokenizer
