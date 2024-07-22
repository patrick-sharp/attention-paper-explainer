import math
import re
import sys

from dataclasses import dataclass

import torch
from torch.nn.functional import softmax

import model
import masking

de_0 = "Der Mann ging zum Markt. Der Weg dorthin dauerte dreißig Minuten. Dieser Satz ist nur ein Füllwort und hat keinen sinnvollen Inhalt. Er kaufte Lebensmittel."
en_0 = "The man went to the market. The journey there took thirty minutes. This sentence is just filler, and has no meaningful content. He bought groceries."

de_1 = "Die Frau ging zum Park. Sie kaufte nichts gekauft."
en_1 = "The woman went to the park. She bought nothing."


def prettify(config, translation):
    """strip the word-delimiting special characters (csp, eow) out of the translation so it looks nicer"""
    translation = translation.replace(" " + config.csp_token, "")
    translation = translation.replace(config.eow_token, "")
    # cleaup spaces between words and punctuation
    # the whitespace pre-tokenizer splits on r"\w+|[^\w\s]+"
    # each match for that regex in the input sentence will be a token
    translation = re.sub(r" ([^\w\s])", r"\1", translation)
    return translation


# a candidate translation in the beam search
@dataclass
class Candidate:
    token_ids: torch.Tensor
    neg_log_ppl: float
    score: float
    in_progress: bool

# a possible finished sentence.
# we keep track of these as we generate predicted tokens.
@dataclass
class Hypothesis:
    token_ids: torch.Tensor
    sum_log_prob: float
    score: float

def score(sum_log_prob, length, length_penalty_alpha):
    """sum_log_prob is the sum of the logs of the probabilities of all predicted tokens.
       length is the length of the generated sequence.
       length_penalty_alpha is a hyperparameter that controls how much longer sequences are prioritized"""
    denominator = ((5.0 + length) / 6.0) ** length_penalty_alpha
    return sum_log_prob / denominator


def _translate_batch_beam_search(ctx, encoder_input, source_mask):
    """Translates using beam search, which will return multiple possible translations.
    Returns list of tuples of (perplexity, translation)

    encoder_input is a tensor of shape (batch_size, seq_len)
    source_mask is a tensor of shape (batch_size, 1, 1, seq_len)"""

    config = ctx.config
    device = config.device
    tokenizer = ctx.tokenizer
    model = ctx.model
    max_translation_extra_tokens = config.max_translation_extra_tokens
    bos_token_id = ctx.bos_token_id
    eos_token_id = ctx.eos_token_id
    pad_token_id = ctx.pad_token_id
    vocab_size = tokenizer.get_vocab_size()
    num_beams = config.num_beams
    length_penalty_alpha = config.length_penalty_alpha

    model.eval()

    batch_size, seq_len = encoder_input.shape

    # (batch_size, seq_len, d_model)
    encoder_output = model.encode(encoder_input, source_mask)
    # since the encoder output and the decoder_input should be the same shape, we
    # repeat the encoder_output.
    # the batch size for the generated encoder_output will be the number of in 
    # progress sentences, so we make num_beams rows of encoder_output per original row.
    encoder_output = encoder_output.repeat_interleave(batch_size * num_beams, 0)
    source_mask = source_mask.repeat_interleave(batch_size * num_beams, 0)

    # keep a list of at most num_beams hypotheses for each sentence in the batch.
    # They start empty because we don't have any finished sentences yet.
    beam_hypotheses = [[]] * batch_size

    # the generated output sentences
    decoder_input = torch.zeros((batch_size * num_beams, 1), dtype=torch.int32, device=device)
    decoder_input.fill_(bos_token_id)

    # (batch_size, num_beams)
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
    # start with negative infinity in all beams except one in order to make sure we
    # don't get identical beams. If we don't do this, then the num_beams next tokens
    # after the bos character will all be the same token. If the next tokens all come
    # from one beam, they will be unique.
    beam_scores[:, 1:] = -math.inf
    beam_scores = beam_scores.view(batch_size * num_beams)

    max_translation_len = seq_len + max_translation_extra_tokens

    all_beams_done = False
    translation_len = 0
    while not all_beams_done and translation_len < max_translation_len:
        translation_len += 1

        # (batch_size * num_beams, 1, dec_seq_len, dec_seq_len)
        target_mask = masking.create_target_mask(decoder_input, pad_token_id)
        # (batch_size * num_beams, dec_seq_len, vocab_size)
        projection = model.decode(
            decoder_input, encoder_output, source_mask, target_mask
        )

        # un-softmaxed logits for the predictions of the next token
        # (batch_size * num_beams, vocab_size)
        logits = projection[:, -1, :]

        # (batch_size * num_beams, vocab_size)
        probabilities = softmax(logits, dim=1)

        # do this so that we can get the top tokens for each beam
        # (batch_size, num_beams * vocab_size)
        probabilities_view = probabilities.view(batch_size, num_beams * vocab_size)

        # we'll need this for getting the top results
        _, vocab_size = probabilities.shape

        # take two tokens so that we always have at least num_beams in progress sentences.
        # if you only take one token, it might be an eos token and then you can't keep
        # using that beam any more.
        topk = torch.topk(probabilities, num_beams * 2, largest=True, sorted=True)
        
        topk_probabilities = topk[0]
        topk_idxs = topk[1]

    #translations = []
    #for cand in candidates:
    #    translation = tokenizer.decode(cand.token_ids[0].tolist())
    #    prettified = prettify(config, translation)
    #    translations.append((cand.neg_log_ppl, prettified))

    #return translations

def translate_beam_search(components, sentence):
    """Translates using beam search, which will return multiple possible translations.
    Returns list of tuples of (perplexity, translation)"""
    config = components.config
    tokenizer = components.tokenizer
    model = components.model
    max_translation_extra_tokens = config.max_translation_extra_tokens
    bos_token_id = components.bos_token_id
    eos_token_id = components.eos_token_id
    pad_token_id = components.pad_token_id
    vocab_size = tokenizer.get_vocab_size()
    num_beams = config.num_beams
    length_penalty_alpha = config.length_penalty_alpha

    model.eval()

    token_ids = tokenizer.encode(sentence).ids

    # add a batch_size dimension of 1
    encoder_input = torch.tensor(token_ids).unsqueeze(0)

    source_mask = masking.create_source_mask(encoder_input, pad_token_id)
    encoder_output = model.encode(encoder_input, source_mask)

    # length normalization from Wu et. al paper
    # increases with length.
    # we use this to give longer sentences a better shot against shorter ones.
    def length_penalty(length, alpha):
        return ((5.0 + length) / 6.0) ** alpha

    # the score by which to judge different candidate translations
    # lower is better
    def compute_score(neg_log_ppl, length):
        lp = length_penalty(length, length_penalty_alpha)
        return neg_log_ppl / lp

    # these are both lists of tuples of (token id tensor, negative log perplexity).
    # negative log perplexity is -log(perplexity)
    # higher perplexity = more confidence
    # higher negative log perplexity = less confidence

    start = torch.empty(1, 1, dtype=torch.int32)
    start.fill_(bos_token_id)

    # start with only one entry so that beams are guaranteed to be unique
    in_progress = [Candidate(start, 0.0, 0.0, True)]
    ended = []

    translation_len = 0
    max_translation_len = len(token_ids) + max_translation_extra_tokens
    while len(in_progress) > 0 and translation_len < max_translation_len:
        translation_len += 1

        tensors = [c.token_ids for c in in_progress]

        # batch_size, dec_seq_len
        decoder_input = torch.cat(tensors, dim=0)

        # the encoder input and source mask are for a batch size of 1.
        # the batch size here will be the number of in progress sentences,
        # so we repeat the encoder output and source mask along the first dimension
        batch_size = len(in_progress)
        encoder_output_batch = encoder_output.expand(batch_size, -1, -1)
        source_mask_batch = source_mask.expand(batch_size, -1, -1, -1)

        # (batch_size, 1, dec_seq_len, dec_seq_len)
        target_mask = masking.create_target_mask(decoder_input, pad_token_id)
        # (batch_size, dec_seq_len, vocab_size)
        projection = model.decode(
            decoder_input, encoder_output_batch, source_mask_batch, target_mask
        )
        # where you left off, just getting the dimension repeat to work

        # un-softmaxed logits for the predictions of the next token
        # (batch_size, vocab_size)
        logits = projection[:, -1, :]

        # (batch_size, vocab_size)
        probabilities = softmax(logits, dim=1)

        # we'll need this for getting the top results
        _, vocab_size = probabilities.shape

        # idxs shape is (num_beams)
        vals, idxs = torch.topk(probabilities.flatten(), num_beams)

        # the beams the predicted token ids are for.
        # (num_beams)
        sentence_idxs = idxs // vocab_size

        # the predicted token ids
        # (num_beams)
        next_token_ids = idxs % vocab_size

        # now we need to select the best candidate sentences from the already ended
        # sentences and the newly predicted topk sentences
        candidates = []
        for i in range(num_beams):
            sentence_idx = sentence_idxs[i]
            next_token_id = next_token_ids[i]

            # append the predicted token onto the sentence
            cand = in_progress[sentence_idx]
            _, dec_seq_len = cand.token_ids.shape
            with_next = torch.empty((1, dec_seq_len + 1), dtype=torch.int32)
            with_next[0, 0:dec_seq_len] = cand.token_ids
            with_next[0, -1] = next_token_id

            # compute the negative log perplexity and the score of the new candidate
            next_token_probability = probabilities[sentence_idx, next_token_id]
            next_neg_log_ppl = cand.neg_log_ppl - math.log(next_token_probability)
            score = compute_score(next_neg_log_ppl, dec_seq_len + 1)
            # True because this is an in progress sentence
            is_in_progress = next_token_id.item() != eos_token_id
            candidates.append(
                Candidate(with_next, next_neg_log_ppl, score, is_in_progress)
            )

        # the ended sentences are candidates too.
        # if any new candidates are better than them, unseat them
        for cand in ended:
            candidates.append(cand)

        # sort by score
        candidates.sort(key=lambda x: x.score)

        # take the top num_beams candidates
        candidates = candidates[0:num_beams]

        # separate in progress from ended
        in_progress = [cand for cand in candidates if cand.in_progress]
        ended = [cand for cand in candidates if not cand.in_progress]

    candidates = in_progress + ended
    candidates.sort(key=lambda x: x.score)

    translations = []
    for cand in candidates:
        translation = tokenizer.decode(cand.token_ids[0].tolist())
        prettified = prettify(config, translation)
        translations.append((cand.neg_log_ppl, prettified))

    return translations


def translate_greedy(components, sentence):
    """Returns a single translation of sentence"""
    config = components.config
    tokenizer = components.tokenizer
    model = components.model
    max_translation_extra_tokens = config.max_translation_extra_tokens
    bos_token_id = components.bos_token_id
    eos_token_id = components.eos_token_id
    pad_token_id = components.pad_token_id
    vocab_size = tokenizer.get_vocab_size()
    num_beams = config.num_beams

    model.eval()

    token_ids = tokenizer.encode(sentence).ids

    # add a batch_size dimension of 1
    encoder_input = torch.tensor(token_ids).unsqueeze(0)

    source_mask = masking.create_source_mask(encoder_input, pad_token_id)
    encoder_output = model.encode(encoder_input, source_mask)
    decoder_input = torch.empty(1, 1, dtype=torch.int32).fill_(bos_token_id)

    neg_log_ppl = 0.0

    max_translation_len = len(token_ids) + max_translation_extra_tokens
    while decoder_input.shape[1] < max_translation_len:
        target_mask = masking.create_target_mask(decoder_input, pad_token_id)

        # (1, dec_seq_len, vocab_size)
        projection = model.decode(
            decoder_input, encoder_output, source_mask, target_mask
        )

        # (1, vocab_size)
        logits = projection[0, -1, :]
        probabilities = softmax(logits, dim=0)

        # get the index of the highest prediction value
        probability, next_token_id = torch.max(probabilities, dim=0)
        next_token_id = next_token_id.item()
        neg_log_ppl -= math.log(probability.item())
        next_token_tensor = torch.empty(1, 1).fill_(next_token_id)

        # (batch_size, dec_seq_len)
        decoder_input = torch.cat(
            [decoder_input, next_token_tensor],
            dim=1,
        ).int()

        if next_token_id == eos_token_id:
            break

    translation = tokenizer.decode(decoder_input[0].tolist())

    return neg_log_ppl, prettify(config, translation)


def print_comparison(sentence, reference_translation, translations):
    """translations should be a tuple of (float, string). The float represents negative log perplexity"""
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


def translate(components, sentence=None, use_beam_search=True):
    reference_translation = None
    if sentence is None:
        sentence = en_0
        reference_translation = de_0

    # must have a tokenizer and a model
    components.require(components.types.TOKENIZER)
    components.require(components.types.MODEL_TRAIN_STATE)

    if use_beam_search:
        tokenizer = components.tokenizer
        pad_token_id = components.pad_token_id
        token_ids = tokenizer.encode(sentence).ids

        # add a batch_size dimension of 1
        encoder_input = torch.tensor(token_ids).unsqueeze(0)

        source_mask = masking.create_source_mask(encoder_input, pad_token_id)
        translations = _translate_batch_beam_search(components, encoder_input, source_mask)
        return
    else:
        translation = [translate_greedy(components, sentence)]

    print_comparison(sentence, reference_translation, translations)
