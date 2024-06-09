import torch


def create_source_mask(encoder_input, pad_token_id):
    # true for padding, false for non-padding
    # (batch_size, seq_len)
    source_mask = encoder_input == pad_token_id

    # (batch_size, 1, 1, seq_len)
    # this allows us to mask all heads of attention at once later
    source_mask = source_mask.unsqueeze(1).unsqueeze(1)
    return source_mask


def create_target_mask(decoder_input, pad_token_id):
    batch_size, seq_len = decoder_input.shape
    # (batch_size, seq_len)
    # true for padding, false for non-padding
    target_pad_mask = decoder_input == pad_token_id

    causal_mask_shape = (
        batch_size,
        1,
        seq_len,
        seq_len,
    )
    # (batch_size, 1, seq_len, seq_len
    # only allows attention to past tokens
    # e.g.
    # [[[[False., True.,  True.],
    #    [False., False., True.],
    #    [False., False., False.]]]]
    target_causal_mask = torch.triu(
        torch.ones(causal_mask_shape, dtype=torch.bool), diagonal=1
    )

    # (batch_size, 1, 1, seq_len)
    target_pad_mask = target_pad_mask.unsqueeze(1).unsqueeze(1)

    # (batch_size, 1, seq_len, seq_len)
    target_mask = target_pad_mask | target_causal_mask
    return target_mask
