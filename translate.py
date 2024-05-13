import sys

import torch

import config
from components import Components
import dataset
import model


def translate(components, sentence):
    config = components.config
    tokenizer = components.tokenizer

    tokens = tokenizer.encode(sentence).ids

    # add a batch_size dimension of 1
    encoder_input = torch.tensor(tokens).unsqueeze(0)

    pad_token = config.pad_token
    pad_token_id = tokenizer.token_to_id(pad_token)
    max_seq_len = config.max_seq_len

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

    while decoder_input.shape[1] < 10:
        target_mask = dataset.create_target_mask(decoder_input, pad_token_id)

        # (batch_size, dec_seq_len, vocab_size)
        projection = model.decode(
            decoder_input, encoder_output, source_mask, target_mask
        )

        # (vocab_size)
        next_token_predictions = projection[0, -1, :]

        # (1)
        # get the index of the highest prediction value
        _, next_token_id = torch.max(next_token_predictions, dim=0)

        # (batch_size, dec_seq_len)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).fill_(next_token_id.item())],
            dim=1,
        ).int()

        if next_token_id == eos_token_id:
            break

    translation = tokenizer.decode(decoder_input[0].tolist())
    return translation


def main(argv):
    if len(sys.argv) == 1:
        # sentence from the wmt14 test set
        sentence = "Im Januar habe ich ein ermutigendes Gespräch mit ihm geführt."
        reference_translation = "In January, I had an encouraging discussion with him."
        print("Translating:")
        print(f'"{sentence}"')
        print()
        print("Reference translation:")
        print(f'"{reference_translation}"')
        print()
        print("Model translation:")
    elif len(sys.argv) == 2:
        sentence = sys.argv[1]
    else:
        print("Too many arguments")
        return

    cmp = Components(config.DEFAULT_CONFIG)

    TOKENIZER = cmp.types.TOKENIZER
    MODEL_TRAIN_STATE = cmp.types.MODEL_TRAIN_STATE

    if not cmp.present[TOKENIZER]:
        cmp.create(TOKENIZER)
    if not cmp.present[MODEL_TRAIN_STATE]:
        cmp.create(MODEL_TRAIN_STATE)

    translation = translate(cmp, sentence)
    print(f"\"{translation}\"")


if __name__ == "__main__":
    main(sys.argv)
