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


def score(log_ppl, length, length_penalty_alpha):
    """log_ppl is the sum of the logs of the probabilities of all predicted tokens
    (the log of the perplexity of a hypothesis).
    length is the length of the generated sequence.
    length_penalty_alpha is a hyperparameter that controls how much longer sequences are prioritized.
    Note that if you give this function zero-dimensional tensors instead of floats, you will get
    a zero-dimensional tensor as output."""
    denominator = ((5.0 + length) / 6.0) ** length_penalty_alpha
    return log_ppl / denominator


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
    # at the end, this will be a list of size batch_size whose elements are lists of
    # size num_beams, whose elements will be tuples of (score: float, hypothesis: torch.Tensor).
    # the beam_hypotheses for each batch are sorted by score in descending order.
    beam_hypotheses = [[]] * batch_size
    # whether all beams for a given sentence are done being translated
    done_translating = [False] * batch_size

    # the worst score for each sentence
    worst_scores = [-math.inf] * batch_size

    # (batch_size * num_beams, seq_len)
    # the generated output sentences.
    # this changes every iteration as we add more predicted tokens
    decoder_input = torch.zeros(
        (batch_size * num_beams, 1), dtype=torch.int32, device=device
    )
    decoder_input.fill_(bos_token_id)

    # (batch_size, num_beams)
    # the sum of the logs of the probabilities of each predicted token for each beam
    # for each sentence. these are negative because they are the sum of the logs of
    # probabilties. probabilities are >=0 and <=1, so their logs are negative.
    # higher is better.
    beam_log_ppls = torch.zeros(
        (batch_size, num_beams), dtype=torch.float, device=device
    )
    # start with negative infinity in all beams except one in order to make sure we
    # don't get identical beams. If we don't do this, then the num_beams next tokens
    # after the bos character will all be the same token. If the next tokens all come
    # from one beam, they will be unique.
    beam_log_ppls[:, 1:] = -math.inf
    # (batch_size * num_beams)
    beam_log_ppls = beam_log_ppls.view(batch_size * num_beams)

    max_seq_len = seq_len + max_translation_extra_tokens

    seq_len = 0
    while not all(done_translating) and seq_len < max_seq_len:
        seq_len += 1

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

        # take two tokens so that we always have at least num_beams sentences that
        # haven't ended. if you only take one token, it might be an eos token and then
        # you can't keep using that beam any more.
        topk = torch.topk(
            probabilities_view, num_beams * 2, dim=1, largest=True, sorted=True
        )

        # (batch_size, num_beams * 2)
        topk_log_probabilities = torch.log(topk[0])

        # (batch_size, num_beams * 2)
        # these will be from 0 to num_beams * vocab_size
        topk_idxs = topk[1]

        # (batch_size, num_beams * 2)
        topk_token_ids = topk_idxs % vocab_size

        # (batch_size, num_beams * 2)
        topk_beam_idxs = topk_idxs = torch.div(
            topk_idxs, vocab_size, rounding_mode="floor"
        )

        next_beam_tokens = torch.zeros(
            (batch_size, num_beams), dtype=torch.int32, device=device
        )
        next_beam_indices = torch.zeros(
            (batch_size, num_beams), dtype=torch.int32, device=device
        )

        for batch_idx in range(batch_size):
            # if self._done[batch_group_idx]:
            #    if self.num_beams < len(self._beam_hyps[batch_group_idx]):
            #        raise ValueError(f"Batch can only be done if at least {self.num_beams} beams have been generated")
            #    if eos_token_id is None or pad_token_id is None:
            #        raise ValueError("Generated beams >= num_beams -> eos_token_id and pad_token have to be defined")
            #    # pad the batch
            #    next_beam_scores[batch_idx, :] = 0
            #    next_beam_tokens[batch_idx, :] = pad_token_id
            #    next_beam_indices[batch_idx, :] = 0
            #    continue

            if done_translating[batch_idx]:
                continue

            # next tokens for this sentence
            beam_idx = 0
            # iterate over the num_beams * 2 candidate tokens in order of highest
            # probability to lowest probability
            for candidate_idx in range(num_beams * 2):
                # from 0.0 to 1.0
                log_probability = topk_log_probabilities[batch_idx, candidate_idx]

                # from 0 to vocab_size
                token_id = topk_token_ids[batch_idx, candidate_idx]

                # index from 0 to num_beams
                beam_idx = topk_beam_idxs[batch_idx, candidate_idx]

                # this is the index of the previous predictions for this beam in the
                # decoder_input tensor
                input_idx = batch_idx * num_beams + beam_idx
                # if we predict the end of sentence, then this is a finished hypothesis.
                # we look at
                if token_id.item() == eos_token_id:
                    # TODO: see if it works to not add hypothesis for rank >= num_beams
                    # this might depend on sign of length penalty
                    beam_log_ppl = beam_log_ppls[input_idx]
                    hyp_log_ppl = beam_log_ppl + log_probability
                    hyp_score = score(hyp_log_ppl, seq_len, length_penalty_alpha).item()

                    if len(beam_hypotheses[batch_idx]) < num_beams:
                        beam = decoder_input[input_idx]
                        hyp = torch.cat([beam, token_id.view(1)])
                        beam_hypotheses[batch_idx].append((hyp_score, hyp))
                    elif hyp_score > worst_scores[batch_idx]:
                        beam = decoder_input[input_idx]
                        hyp = torch.cat([beam, token_id.view(1)])
                        # (seq_len)
                        beam_hypotheses[batch_idx][num_beams - 1] = (hyp_score, hyp)
                        beam_hypotheses[batch_idx].sort(
                            key=lambda x: x[0], reverse=True
                        )
                else:
                    # TODO: add else branch for continuing prediction.
                    # prepare the next iteration's sequences
                    pass
        break

        #######################################
        # FRONTIER
        #######################################
        #    for beam_token_rank, (next_token, next_score, next_index) in enumerate(
        #        zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
        #    ):
        #        batch_beam_idx = batch_idx * self.group_size + next_index
        #        # add to generated hypotheses if end of sentence
        #        if (eos_token_id is not None) and (next_token.item() in eos_token_id):
        #            # if beam_token does not belong to top num_beams tokens, it should not be added
        #            is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
        #            if is_beam_token_worse_than_top_num_beams:
        #                continue
        #            if beam_indices is not None:
        #                beam_index = beam_indices[batch_beam_idx]
        #                beam_index = beam_index + (batch_beam_idx,)
        #            else:
        #                beam_index = None

        #            self._beam_hyps[batch_group_idx].add(
        #                input_ids[batch_beam_idx].clone(),
        #                next_score.item(),
        #                beam_indices=beam_index,
        #                generated_len=cur_len - decoder_prompt_len,
        #            )
        #        else:
        #            # add next predicted token since it is not eos_token
        #            next_beam_scores[batch_idx, beam_idx] = next_score
        #            next_beam_tokens[batch_idx, beam_idx] = next_token
        #            next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
        #            beam_idx += 1

        #        # once the beam for next step is full, don't add more tokens to it.
        #        if beam_idx == self.group_size:
        #            break

        #    if beam_idx < self.group_size:
        #        raise ValueError(
        #            f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id:"
        #            f" {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
        #        )

        #    # Check if we are done so that we can save a pad step if all(done)
        #    self._done[batch_group_idx] = self._done[batch_group_idx] or self._beam_hyps[batch_group_idx].is_done(
        #        next_scores[batch_idx].max().item(), cur_len, decoder_prompt_len
        #    )

        # return UserDict(
        #    {
        #        "next_beam_scores": next_beam_scores.view(-1),
        #        "next_beam_tokens": next_beam_tokens.view(-1),
        #        "next_beam_indices": next_beam_indices.view(-1),
        #    }

    # translations = []
    # for cand in candidates:
    #    translation = tokenizer.decode(cand.token_ids[0].tolist())
    #    prettified = prettify(config, translation)
    #    translations.append((cand.neg_log_ppl, prettified))

    # return translations


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
        translations = _translate_batch_beam_search(
            components, encoder_input, source_mask
        )
        return
    else:
        translation = [translate_greedy(components, sentence)]

    print_comparison(sentence, reference_translation, translations)
