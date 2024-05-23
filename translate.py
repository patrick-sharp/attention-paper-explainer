import math
import sys

import torch
from torch.nn.functional import softmax

from configuration import DEFAULT_CONFIG
import components
import dataset
import model

de_0 = "Wiederaufnahme der Sitzungsperiode"
en_0 = "Resumption of the session"


def beam_search(components, encoder_input, source_mask, prettify=True):
    """Translates using beam search, which will return multiple possible translations."""
    config = components.config
    tokenizer = components.tokenizer
    model = components.model
    max_translation_len = config.max_translation_len
    bos_token = config.bos_token
    eos_token = config.eos_token
    pad_token = config.pad_token
    csp_token = config.csp_token
    eow_token = config.eow_token
    vocab_size = tokenizer.get_vocab_size()
    beam_width = config.beam_width
    length_penalty_alpha = config.length_penalty_alpha

    bos_token_id = tokenizer.token_to_id(bos_token)
    eos_token_id = tokenizer.token_to_id(eos_token)
    pad_token_id = tokenizer.token_to_id(pad_token)
    model.eval()

    encoder_output = model.encode(encoder_input, source_mask)

    # length normalization from Wu et. al paper
    def lp(length, alpha):
        return ((5.0 + length) / 6.0) ** alpha

    translation_len = 0
    # negative log perplexity
    perplexities = [0.0 for _ in range(beam_width)]
    decoder_inputs = []
    for _ in range(beam_width):
        start = torch.empty(1, 1, dtype=torch.int32)
        start.fill_(bos_token_id)
        decoder_inputs.append(start)

    ended_perplexities = []
    ended_decoder_inputs = []

    while len(decoder_inputs) > 0 and translation_len < max_translation_len:
        translation_len += 1

        # compute predictions for each item in the beam
        beam_probabilities = []
        for decoder_input in decoder_inputs:
            target_mask = dataset.create_target_mask(decoder_input, pad_token_id)

            # (1, dec_seq_len, vocab_size)
            projection = model.decode(
                decoder_input, encoder_output, source_mask, target_mask
            )

            # (vocab_size)
            logits = projection[0, -1, :]

            # (vocab_size)
            probabilities = softmax(logits, dim=0)
            beam_probabilities.append(probabilities)

        # scores(prev_ppl, pred, length) = (prev_ppl + ln(pred))/lp(length)
        length_penalty = lp(translation_len, length_penalty_alpha)
        beam_scores = []
        for i in range(len(decoder_inputs)):
            ppl = perplexities[i]
            probabilities = beam_probabilities[i]
            scores = ppl - torch.log(probabilities)
            scores /= length_penalty
            scores = scores.unsqueeze(-1) # so we can cat them on dim 1
            beam_scores.append(scores)
        # (vocab_size, len(decoder_inputs))
        beam_scores = torch.cat(beam_scores, dim=1)
        # (vocab_size)
        superscored, superscored_i = torch.min(beam_scores, dim=1)
        vals, inds = torch.topk(
            superscored, len(decoder_inputs), largest=False
        )

        next_decoder_inputs = []
        next_perplexities = []
        for token_id in inds.tolist():
            beam_i = superscored_i[i]
            next_token_tensor = torch.empty(1, 1).fill_(token_id)
            decoder_input = decoder_inputs[beam_i]
            decoder_input = torch.cat(
                [decoder_input, next_token_tensor],
                dim=1,
            ).int()
            prev_perplexity = perplexities[beam_i]
            probabilities = beam_probabilities[beam_i]
            probability = probabilities[token_id].item()
            perplexity = prev_perplexity - math.log(probability)
            if token_id == eos_token_id or translation_len == max_translation_len:
                ended_decoder_inputs.append(decoder_input)
                ended_perplexities.append(perplexity)
            else:
                next_decoder_inputs.append(decoder_input)
                next_perplexities.append(perplexity)
        decoder_inputs = next_decoder_inputs
        perplexities = next_perplexities

    translations = []
    for ppl, x in zip(ended_perplexities, ended_decoder_inputs):
        x = tokenizer.decode(x[0].tolist())
        if prettify:
            x = x.replace(" " + csp_token, "")
            x = x.replace(eow_token, "")
            x = x.replace(csp_token, "")
        translations.append((ppl, x))

    # sort by perplexity so we show the best guesses first
    def sort_ppl(x):
        return x[0]
    return list(sorted(translations, key=sort_ppl))


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


def translate_beam(components, sentence, prettify=True):
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
    return beam_search(components, encoder_input, source_mask)


def print_comparison(sentence, reference_translation, translations):
    print("Translating:")
    print(f'"{sentence}"')
    print()
    if reference_translation is not None:
        print("Reference translation:")
        print(f'"{reference_translation}"')
        print()
    print("Model translations:")
    for ppl, x in translations:
        print(f'{ppl: 7.3f} "{x}"')


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

    # translation = translate_sentence(cmp, sentence)
    # print_comparison(sentence, reference_translation, translation)
    translations = translate_beam(cmp, sentence)
    print_comparison(sentence, reference_translation, translations)


if __name__ == "__main__":
    main(sys.argv)
