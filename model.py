import torch
import torch.nn as nn
import torch.nn.functional as functional

import math


# Note: no dropout. Dropout is applied to the sum of the embedding and
# positional encoding
class ScaledEmbedding(nn.Module):
    def __init__(self, components):
        super().__init__()
        config = components.config
        vocab_size = components.tokenizer.get_vocab_size()
        self.d_model = config.d_model
        self.embedding = nn.Embedding(vocab_size, config.d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.p_dropout)

        # (max_seq_len, d_model)
        positional_encodings = torch.zeros(config.max_seq_len, config.d_model)

        # (max_seq_len)
        position = torch.arange(0, config.max_seq_len, dtype=torch.float)

        # (max_seq_len, 1)
        position = position.unsqueeze(1)

        # (d_model / 2)
        evens = torch.arange(0, config.d_model, 2).float()

        # (d_model / 2)
        exponent = evens / config.d_model

        # (d_model / 2)
        denominator = torch.pow(torch.ones(config.d_model // 2) * 10000.0, exponent)

        # sine even indices
        positional_encodings[:, 0::2] = torch.sin(position / denominator)

        # cosine odd indices
        positional_encodings[:, 1::2] = torch.cos(position / denominator)

        # (1, max_seq_len, d_model)
        positional_encodings = positional_encodings.unsqueeze(0)

        # Register the positional encoding as a buffer
        self.register_buffer("positional_encodings", positional_encodings)

    def forward(self, x):
        _, seq_len, _ = x.shape
        # broadcasts addition over all sequences in batch
        # (batch_size, seq_len, d_model)
        x = x + self.positional_encodings[:, :seq_len, :]

        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, config, stack=None, layer=None):
        super().__init__()

        # used for the instrumentation for making attention plots
        self.stack = stack
        self.layer = layer

        # we'll need some values from here in the forward pass
        self.config = config

        # so we don't have to type out config.* a bunch of times
        d_model = config.d_model
        d_key = config.d_key
        d_value = config.d_value
        num_heads = config.num_heads
        bias = config.bias
        p_dropout = config.p_dropout

        # (d_model, num_heads * d_key)
        self.w_q = nn.Linear(d_model, num_heads * d_key, bias=bias)

        # (d_model, num_heads * d_key)
        self.w_k = nn.Linear(d_model, num_heads * d_key, bias=bias)

        # (d_model, num_heads * d_value)
        self.w_v = nn.Linear(d_model, num_heads * d_value, bias=bias)

        # combines all heads
        # (num_heads * d_value, d_model)
        self.w_o = nn.Linear(num_heads * d_value, d_model, bias=bias)

        # in Karpathy's tutorial, he uses two dropouts.
        # The paper seems to only imply one (at the end of each sub-layer).
        # not sure why there's a difference, but there you go.
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, query, key, value, mask, instrumentation=None):
        # For encoder self attention:
        # query, key, and value are all the same
        # mask is the source mask
        #   (batch_size, 1, 1, enc_seq_len)

        # For decoder masked self attention:
        # query, key, and value are all the same
        # mask is the target mask
        #   (batch_size, 1, dec_seq_len, dec_seq_len)

        # For decoder cross attention:
        # query and key are the same (the encoder output)
        # value is the input from the previous sublayer
        # mask is the source mask
        #   (batch_size, 1, 1, enc_seq_len)

        # the key and value must have the same sequence length, but the query
        # can have a different one. This comes in handy during translation
        # because you only generate one token at a time. This saves a lot of
        # compute by not having to pad both sequences to the same length.
        batch_size, q_seq_len, d_model = query.shape
        _, k_seq_len, _ = value.shape
        _, v_seq_len, _ = value.shape
        # assert v_seq_len == k_seq_len

        d_key = self.config.d_key
        d_value = self.config.d_value
        num_heads = self.config.num_heads
        p_dropout = self.config.p_dropout

        # calculate the queries, keys, and values for all heads at once
        # INTERMISSION: learn about the rules for batched matrix multiplies
        # https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
        # https://pytorch.org/docs/stable/generated/torch.matmul.html

        # (batch_size, q_seq_len, num_heads * d_key)
        q = self.w_q(query)

        # (batch_size, v_seq_len, num_heads * d_key)
        k = self.w_k(key)

        # (batch_size, v_seq_len, num_heads * d_value)
        v = self.w_v(value)

        # we want all the queries * all the keys (dot prod) for each head
        # we want (batch_size, num_heads, q_seq_len, seq_len
        #
        # get one score for one token:
        #    dot product of query for token and key for token
        #    (1, d_key)
        #    @ (d_key, 1)
        #    = (1, 1)
        #
        # get all the scores for a token
        #    dot product of query for token and all keys for token
        #    (1, d_key)
        #    @ (d_key, q_seq_len)
        #    = (1, q_seq_len)
        #
        # get scores for one head
        #    (q_seq_len, d_key)
        #    @ (d_key, k_seq_len)
        #    = (q_seq_len, k_seq_len)
        #
        # INTERMISSION: remind about matmul rules
        #
        # get the scores for all heads for one sentence in the batch
        #    (num_heads, q_seq_len, d_key)
        #    @ (num_heads, d_key, k_seq_len)
        #    = (num_heads, seq_len, k_seq_len)
        #
        # get the scores for all sentences in a batch
        #    (batch_size, num_heads, q_seq_len, d_key)
        #    @ (batch_size, num_heads, d_key, k_seq_len)
        #    = (batch_size, num_heads, q_seq_len, k_seq_len)
        #
        # INTERMISSION: learn about how torch.view works and the underlying
        #    order of tensor in memory

        # split into heads

        # (batch_size, q_seq_len, num_heads, d_key)
        q = q.view(batch_size, q_seq_len, num_heads, d_key)
        # (batch_size, num_heads, q_seq_len, d_key)
        q = q.transpose(1, 2)

        # (batch_size, k_seq_len, num_heads, d_key)
        k = k.view(batch_size, k_seq_len, num_heads, d_key)
        # (batch_size, num_heads, k_seq_len, d_key)
        k = k.transpose(1, 2)
        # (batch_size, num_heads, d_key, k_seq_len)
        kT = k.transpose(2, 3)

        # (batch_size, k_seq_len, num_heads, d_value)
        v = v.view(batch_size, k_seq_len, num_heads, d_value)
        # (batch_size, num_heads, k_seq_len, d_value)
        v = v.transpose(1, 2)

        # compute attention scores
        # (batch_size, num_heads, q_seq_len, k_seq_len)
        x = q @ kT
        # scale attention scores
        x = x / math.sqrt(d_key)
        # mask so that the model won't pay attention to padding
        x.masked_fill_(mask, -math.inf)
        # softmax to get probabilities
        x = functional.softmax(x, dim=-1)
        # right here is where karpathy would insert the other dropout

        # if we passed in an instrumentation map to the model, store
        # attention scores in it for later plotting
        if (
            self.stack == "encoder"
            and instrumentation is not None
            and instrumentation["layer"] == self.layer
        ):
            head = instrumentation["head"]
            # (seq_len, seq_len)
            attention = x[0, head, :, :].detach()
            instrumentation["attention"] = attention

        # multiply the attention scores by the value vectors and sum them
        # (batch_size, num_heads, q_seq_len, d_value)
        x = x @ v

        # re-combine the data from all the heads
        # (batch_size, q_seq_len, num_heads, d_value)
        x = x.transpose(1, 2)

        # tensor is not contiguous because it was created by taking the transpose
        # of a different tensor.
        # use .contiguous to force it into being contiguous (i.e. ordered) in memory.
        # (batch_size, q_seq_len, num_heads * d_value)
        x = x.contiguous().view(batch_size, q_seq_len, num_heads * d_value)

        # output projection
        # note that this has the same dimension as the original input
        # (batch_size, q_seq_len, d_model)
        x = self.w_o(x)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config.d_model
        d_ff = config.d_ff
        p_dropout = config.p_dropout
        bias = config.bias

        # (d_model, d_ff)
        self.w_1 = nn.Linear(d_model, d_ff, bias=bias)
        # (d_ff, d_model)
        self.w_2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        # x is (batch_size, seq_len, d_model)
        # (batch_size, seq_len, d_ff)
        x = self.w_1(x)
        # activation function
        x = torch.relu(x)

        # (batch_size, seq_len, d_model)
        x = self.w_2(x)
        x = self.dropout(x)

        return x


class LayerNorm(nn.Module):
    """This is based on the equation shown in the documentation for torch.nn.LayerNorm"""

    def __init__(self, config) -> None:
        super().__init__()
        d_model = config.d_model
        bias = config.bias

        self.layer_norm_epsilon = config.layer_norm_epsilon

        # initialize gamma to ones and bias to zero.
        # this is common practice, since we want to assume that we don't need to much
        # adjustment by default. multiplying by 1 and adding 0 don't affect the normalized values,
        # but the parameters can change during the learning process if it helps.

        # (d_model)
        self.gamma = nn.Parameter(torch.ones(d_model))

        # (d_model)
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # x is (batch_size, seq_len, d_model)

        layer_norm_epsilon = self.layer_norm_epsilon

        # use keepdim to avoid squeezing the last dimension.
        # without keepdim, would be (batch_size, seq_len)
        # (batch_size, seq_len, 1)
        mean = x.mean(dim=-1, keepdim=True)

        # (batch_size, seq_len, 1)
        variance = x.var(dim=-1, keepdim=True)

        # normalize
        # (batch_size, seq_len, d_model
        x = (x - mean) / torch.sqrt(variance + layer_norm_epsilon)

        # adjust
        # (batch_size, seq_len, d_model)
        x = x * self.gamma + self.beta

        return x


class EncoderBlock(nn.Module):
    def __init__(self, config, layer=None):
        super().__init__()
        self.attention = MultiHeadAttention(config, stack="encoder", layer=layer)
        self.layer_norm_1 = LayerNorm(config)

        self.feed_forward = FeedForward(config)
        self.layer_norm_2 = LayerNorm(config)

    def forward(self, x, mask, instrumentation=None):
        residual = x
        x = self.attention(x, x, x, mask, instrumentation)
        x = self.layer_norm_1(x + residual)

        residual = x
        x = self.feed_forward(x)
        x = self.layer_norm_2(x + residual)
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        blocks = [EncoderBlock(config, i) for i in range(config.num_blocks)]
        self.layers = nn.ModuleList(blocks)

    def forward(self, x, mask, instrumentation):
        # x is (batch_size, seq_len, d_model)
        # mask is (batch_size, 1, 1, seq_len)

        for layer in self.layers:
            # (batch_size, seq_len, d_model)
            x = layer(x, mask, instrumentation)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.causal_attention = MultiHeadAttention(config)
        self.layer_norm_1 = LayerNorm(config)

        self.cross_attention = MultiHeadAttention(config)
        self.layer_norm_2 = LayerNorm(config)

        self.feed_forward = FeedForward(config)
        self.layer_norm_3 = LayerNorm(config)

    def forward(self, x, encoder_output, source_mask, target_mask):
        residual = x
        x = self.causal_attention(x, x, x, target_mask)
        x = self.layer_norm_1(x + residual)

        residual = x
        x = self.cross_attention(x, encoder_output, encoder_output, source_mask)
        x = self.layer_norm_2(x + residual)

        residual = x
        x = self.feed_forward(x)
        x = self.layer_norm_3(x + residual)
        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        blocks = [DecoderBlock(config) for _ in range(config.num_blocks)]

        self.layers = nn.ModuleList(blocks)

    def forward(self, x, encoder_output, source_mask, target_mask):
        # x is (batch_size, seq_len, d_model)
        # source_mask is (batch_size, 1, 1, seq_len)
        # target_mask is (batch_size, 1, seq_len, seq_len)

        for layer in self.layers:
            # (batch_size, seq_len, d_model)
            x = layer(x, encoder_output, source_mask, target_mask)

        return x


class ProjectionLayer(nn.Module):
    def __init__(self, components):
        super().__init__()
        config = components.config
        d_model = config.d_model
        vocab_size = components.tokenizer.get_vocab_size()
        bias = config.bias

        self.linear = nn.Linear(d_model, vocab_size, bias=bias)

    def forward(self, x):
        # x is (batch_size, seq_len, d_model)

        # Note that we don't apply softmax here. The loss functions used in training
        # expect un-normalized logits

        # (batch_size, seq_len, vocab_size)
        return self.linear(x)


class Transformer(nn.Module):
    def __init__(self, components):
        super().__init__()
        config = components.config
        self.config = config

        self.source_embedding = ScaledEmbedding(components)
        self.positional_encoding = PositionalEncoding(config)
        self.encoder = Encoder(config)

        self.target_embedding = ScaledEmbedding(components)
        self.decoder = Decoder(config)

        self.projection_layer = ProjectionLayer(components)
        self.to(components.device)

    def encode(self, encoder_input, mask, instrumentation=None):
        x = self.source_embedding(encoder_input)
        x = self.positional_encoding(x)
        x = self.encoder(x, mask, instrumentation)
        return x

    def decode(self, decoder_input, encoder_output, source_mask, target_mask):
        x = self.target_embedding(decoder_input)
        x = self.positional_encoding(x)
        x = self.decoder(x, encoder_output, source_mask, target_mask)

        x = self.projection_layer(x)
        return x

    def forward(self, encoder_input, decoder_input, source_mask, target_mask):
        """training forward pass, not translation.
        encoder_input: (batch_size, enc_seq_len)
        decoder_input: (batch_size, dec_seq_len)
        source_mask: (batch_size, 1, 1, enc_seq_len)
        target_mask: (batch_size, 1, dec_seq_len, dec_seq_len)
        """

        assert len(encoder_input.shape) == 2
        batch_size, enc_seq_len = encoder_input.shape
        _, dec_seq_len = decoder_input.shape
        assert decoder_input.shape == (batch_size, dec_seq_len)
        assert source_mask.shape == (batch_size, 1, 1, enc_seq_len)
        assert target_mask.shape == (batch_size, 1, dec_seq_len, dec_seq_len)

        x = self.encode(encoder_input, source_mask)

        encoder_output = x
        x = self.decode(decoder_input, encoder_output, source_mask, target_mask)
        return x
