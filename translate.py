import sys

import torch

from configuration import DEFAULT_CONFIG
import components
import dataset
import model

de_0 = "Wiederaufnahme der Sitzungsperiode"
en_0 = "Resumption of the session"


def translate_tensor(components, encoder_input, source_mask, prettify=True):
    config = components.config
    tokenizer = components.tokenizer
    model = components.model
    model.eval()

    encoder_output = model.encode(encoder_input, source_mask)

    bos_token = config.bos_token
    bos_token_id = tokenizer.token_to_id(bos_token)

    eos_token = config.eos_token
    eos_token_id = tokenizer.token_to_id(eos_token)

    pad_token = config.pad_token
    pad_token_id = tokenizer.token_to_id(pad_token)

    max_translation_len = config.max_translation_len
    decoder_input = torch.empty(1, 1, dtype=torch.int32).fill_(bos_token_id)

    while decoder_input.shape[1] < max_translation_len:
        target_mask = dataset.create_target_mask(decoder_input, pad_token_id)

        # (1, dec_seq_len, vocab_size)
        projection = model.decode(
            decoder_input, encoder_output, source_mask, target_mask
        )

        # (1, vocab_size)
        next_token_logits = projection[0, -1, :]

        # get the index of the highest prediction value
        _, next_token_id = torch.max(next_token_logits, dim=0)
        next_token_id = next_token_id.item()
        next_token_tensor = torch.empty(1, 1).fill_(next_token_id)

        # (batch_size, dec_seq_len)
        decoder_input = torch.cat(
            [decoder_input, next_token_tensor],
            dim=1,
        ).int()

        if next_token_id == eos_token_id:
            break

    translation = tokenizer.decode(decoder_input[0].tolist())
    if prettify:
        # strip the word-delimiting special characters out of the translation so it looks nicer
        translation = translation.replace(" " + DEFAULT_CONFIG.csp_token, "")
        translation = translation.replace(DEFAULT_CONFIG.eow_token, "")
        translation = translation.replace(DEFAULT_CONFIG.csp_token, "")

    return translation


def translate_sentence(components, sentence, prettify=True):
    config = components.config
    tokenizer = components.tokenizer

    tokens = tokenizer.encode(sentence).ids

    # add a batch_size dimension of 1
    encoder_input = torch.tensor(tokens).unsqueeze(0)

    pad_token = config.pad_token
    pad_token_id = tokenizer.token_to_id(pad_token)
    max_translation_len = config.max_translation_len

    model = components.model

    # switch to eval mode
    components.model.eval()

    source_mask = dataset.create_source_mask(encoder_input, pad_token_id)

    bos_token = config.bos_token
    bos_token_id = tokenizer.token_to_id(bos_token)

    encoder_output = model.encode(encoder_input, source_mask)
    # decoder_input begins as just a bos_token
    decoder_input = torch.empty(1, 1, dtype=torch.int32).fill_(bos_token_id)

    eos_token = config.eos_token
    eos_token_id = tokenizer.token_to_id(eos_token)

    while decoder_input.shape[1] < max_translation_len:
        target_mask = dataset.create_target_mask(decoder_input, pad_token_id)

        # (batch_size, dec_seq_len, vocab_size)
        projection = model.decode(
            decoder_input, encoder_output, source_mask, target_mask
        )

        # (vocab_size)
        next_token_logits = projection[0, -1, :]

        # get the index of the highest prediction value
        _, next_token_id = torch.max(next_token_logits, dim=0)
        next_token_id = next_token_id.item()
        next_token_tensor = torch.empty(1, 1).fill_(next_token_id)

        # (batch_size, dec_seq_len)
        decoder_input = torch.cat(
            [decoder_input, next_token_tensor],
            dim=1,
        ).int()

        if next_token_id == eos_token_id:
            break

    translation = tokenizer.decode(decoder_input[0].tolist())
    if prettify:
        # strip the word-delimiting special characters out of the translation so it looks nicer
        translation = translation.replace(" " + DEFAULT_CONFIG.csp_token, "")
        translation = translation.replace(DEFAULT_CONFIG.eow_token, "")
        translation = translation.replace(DEFAULT_CONFIG.csp_token, "")

    return translation


def print_comparison(sentence, reference_translation, translation):
    print("Translating:")
    print(f'"{sentence}"')
    print()
    if reference_translation is not None:
        print("Reference translation:")
        print(f'"{reference_translation}"')
        print()
    print("Model translation:")
    print(f'"{translation}"')


def main(argv):
    if len(sys.argv) == 1:
        # sentence from the wmt14 test set
        sentence = de_0
        reference_translation = en_0
    elif len(sys.argv) == 2:
        sentence = sys.argv[1]
    else:
        print("Too many arguments")
        return

    cmp = components.Components(DEFAULT_CONFIG)

    TOKENIZER = cmp.types.TOKENIZER
    MODEL_TRAIN_STATE = cmp.types.MODEL_TRAIN_STATE

    if not cmp.present[TOKENIZER]:
        cmp.create(TOKENIZER)
    if not cmp.present[MODEL_TRAIN_STATE]:
        cmp.create(MODEL_TRAIN_STATE)

    translation = translate_sentence(cmp, sentence)
    print_comparison(sentence, reference_translation, translation)


if __name__ == "__main__":
    main(sys.argv)
