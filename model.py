import torch
import torch.nn as nn
import torch.nn.functional as functional

import math

from config import DEFAULT_CONFIG


# Note: no dropout. Dropout is applied to the sum of the embedding and
# positional encoding
class ScaledEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.p_dropout)

        # (max_sequence_length, d_model)
        positional_encodings = torch.zeros(config.max_sequence_length, config.d_model)

        # (max_sequence_length)
        position = torch.arange(0, config.max_sequence_length, dtype=torch.float)

        # (max_sequence_length, 1)
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

        # (1, max_sequence_length, d_model)
        positional_encodings = positional_encodings.unsqueeze(0)

        # Register the positional encoding as a buffer
        self.register_buffer("positional_encodings", positional_encodings)

    def forward(self, x):
        _, sequence_length, _ = x.shape
        # broadcasts addition over all sequences in batch
        # (batch_size, sequence_length, d_model)
        x = x + self.positional_encodings[:, :sequence_length, :]

        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

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
        self.w_q_dropout = nn.Dropout(p_dropout)

        # (d_model, num_heads * d_key)
        self.w_k = nn.Linear(d_model, num_heads * d_key, bias=bias)
        self.w_k_dropout = nn.Dropout(p_dropout)

        # (d_model, num_heads * d_value)
        self.w_v = nn.Linear(d_model, num_heads * d_value, bias=bias)
        self.w_v_dropout = nn.Dropout(p_dropout)

        # combines all heads
        # (num_heads * d_value, d_model)
        self.w_o = nn.Linear(num_heads * d_value, d_model, bias=bias)

        # in Karpathy's tutorial, he uses two dropouts.
        # The paper seems to only imply one (at the end of each sub-layer).
        # not sure why there's a difference, but there you go.
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, query, key, value, mask):
        # For encoder self attention:
        # query, key, and value are all the same
        # mask is the source mask
        #   (batch_size, 1, 1, sequence_length)

        # For decoder masked self attention:
        # query, key, and value are all the same
        # mask is the target mask
        #   (batch_size, 1, sequence_length, sequence_length)

        # For decoder cross attention:
        # query and key are the same (the encoder output)
        # value is the input from the previous sublayer
        # mask is the source mask
        #   (batch_size, 1, 1, sequence_length)

        batch_size, sequence_length, d_model = query.shape

        d_key = self.config.d_key
        d_value = self.config.d_value
        num_heads = self.config.num_heads
        bias = self.config.bias
        p_dropout = self.config.p_dropout

        # calculate the queries, keys, and values for all heads at once
        # INTERMISSION: learn about the rules for batched matrix multiplies
        # https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
        # https://pytorch.org/docs/stable/generated/torch.matmul.html

        # (batch_size, sequence_length, num_heads * d_key)
        q = self.w_q(query)

        # (batch_size, sequence_length, num_heads * d_key)
        k = self.w_k(key)

        # (batch_size, sequence_length, num_heads * d_value)
        v = self.w_v(value)

        # we want all the queries * all the keys (dot prod) for each head
        # we want (batch_size, num_heads, sequence_length, sequence_length
        #
        # get one score for one token:
        #    dot product of query for token and key for token
        #    (1, d_key)
        #    @ (d_key, 1)
        #    = (1, 1)
        #
        # get all the scores for a token
        #    dot product of query for token and key for token
        #    (1, d_key)
        #    @ (d_key, sequence_length)
        #    = (1, sequence_length)
        #
        # get scores for one head
        #    (sequence_length, d_key)
        #    @ (d_key, sequence_length)
        #    = (sequence_length, sequence_length)
        #
        # INTERMISSION: remind about matmul rules
        #
        # get the scores for all heads for one sentence in the batch
        #    (num_heads, sequence_length, d_key)
        #    @ (num_heads, d_key, sequence_length)
        #    = (num_heads, sequence_length, sequence_length)
        #
        # get the scores for all sentences in a batch
        #    (batch_size, num_heads, sequence_length, d_key)
        #    @ (batch_size, num_heads, d_key, sequence_length)
        #    = (batch_size, num_heads, sequence_length, sequence_length)
        #
        # INTERMISSION: learn about how torch.view works and the underlying
        #    order of tensor in memory

        # split into heads

        # (batch_size, sequence_length, num_heads, d_key)
        q = q.view(batch_size, sequence_length, num_heads, d_key)
        # (batch_size, num_heads, sequence_length, d_key)
        q = q.transpose(1, 2)

        # (batch_size, sequence_length, num_heads, d_key)
        k = k.view(batch_size, sequence_length, num_heads, d_key)
        # (batch_size, num_heads, sequence_length, d_key)
        k = k.transpose(1, 2)
        # (batch_size, num_heads, d_key, sequence_length)
        kT = k.transpose(2, 3)

        # (batch_size, sequence_length, num_heads, d_value)
        v = v.view(batch_size, sequence_length, num_heads, d_value)
        # (batch_size, num_heads, sequence_length, d_value)
        v = v.transpose(1, 2)

        # compute attention scores
        # (batch_size, num_heads, sequence_length, sequence_length)
        x = q @ kT
        # scale attention scores
        x = x / math.sqrt(d_key)
        # mask so that the model won't pay attention to padding
        x.masked_fill_(mask, -math.inf)
        # softmax to get probabilities
        x = functional.softmax(x, dim=-1)
        # right here is where karpathy would insert the other dropout

        # multiply the attention scores by the value vectors and sum them
        # (batch_size, num_heads, sequence_length, d_value)
        x = x @ v

        # re-combine the data from all the heads
        # (batch_size, sequence_length, num_heads, d_value)
        x = x.transpose(1, 2)

        # tensor is not contiguous because it was created by taking the transpose
        # of a different tensor.
        # use .contiguous to force it into being contiguous (i.e. ordered) in memory.
        # (batch_size, sequence_length, num_heads * d_value)
        x = x.contiguous().view(batch_size, sequence_length, num_heads * d_value)

        # output projection
        # note that this has the same dimension as the original input
        # (batch_size, sequence_length, d_model)
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
        # x is (batch_size, sequence_length, d_model)
        # (batch_size, sequence_length, d_ff)
        x = self.w_1(x)
        # activation function
        x = torch.relu(x)

        # (batch_size, sequence_length, d_model)
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
        # x is (batch_size, sequence_length, d_model)

        layer_norm_epsilon = self.layer_norm_epsilon

        # use keepdim to avoid squeezing the last dimension.
        # without keepdim, would be (batch_size, sequence_length)
        # (batch_size, sequence_length, 1)
        mean = x.mean(dim=-1, keepdim=True)

        # (batch_size, sequence_length, 1)
        variance = x.var(dim=-1, keepdim=True)

        # normalize
        # (batch_size, sequence_length, d_model
        x = (x - mean) / torch.sqrt(variance + layer_norm_epsilon)

        # adjust
        # (batch_size, sequence_length, d_model)
        x = x * self.gamma + self.beta

        return x


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layer_norm_1 = LayerNorm(config)

        self.feed_forward = FeedForward(config)
        self.layer_norm_2 = LayerNorm(config)

    def forward(self, x, mask):
        x = self.attention(x, x, x, mask)
        x = self.layer_norm_1(x)

        x = self.feed_forward(x)
        x = self.layer_norm_2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        blocks = [EncoderBlock(config) for _ in range(config.num_blocks)]
        self.layers = nn.ModuleList(blocks)

    def forward(self, x, mask):
        # x is (batch_size, sequence_length, d_model)
        # mask is (batch_size, 1, 1, sequence_length)

        for layer in self.layers:
            # (batch_size, sequence_length, d_model)
            x = layer(x, mask)

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
        x = self.causal_attention(x, x, x, target_mask)
        x = self.layer_norm_1(x)

        x = self.cross_attention(encoder_output, encoder_output, x, source_mask)
        x = self.layer_norm_1(x)

        x = self.feed_forward(x)
        x = self.layer_norm_3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        blocks = [DecoderBlock(config) for _ in range(config.num_blocks)]
        self.layers = nn.ModuleList(blocks)

    def forward(self, x, encoder_output, source_mask, target_mask):
        # x is (batch_size, sequence_length, d_model)
        # source_mask is (batch_size, 1, 1, sequence_length)
        # target_mask is (batch_size, 1, sequence_length, sequence_length)

        for layer in self.layers:
            # (batch_size, sequence_length, d_model)
            x = layer(x, encoder_output, source_mask, target_mask)

        return x


class ProjectionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config.d_model
        vocab_size = config.vocab_size
        bias = config.bias

        self.linear = nn.Linear(d_model, vocab_size, bias=bias)

    def forward(self, x):
        # x is (batch_size, sequence_length, d_model)

        # Note that we don't apply softmax here. The loss functions used in training
        # expect un-normalized logits

        # (batch_size, sequence_length, vocab_size)
        return self.linear(x)


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.source_embedding = ScaledEmbedding(config)
        self.positional_encoding = PositionalEncoding(config)
        self.encoder = Encoder(config)

        self.target_embedding = ScaledEmbedding(config)
        self.decoder = Decoder(config)

        self.projection_layer = ProjectionLayer(config)

    def forward(self, encoder_input, decoder_input, source_mask, target_mask):
        """training forward pass, not translation.
        encoder_input: (batch_size, sequence_length, d_model)
        decoder_input: (batch_size, sequence_length, d_model)
        source_mask:  (batch_size, 1, 1, sequence_length)
        target_mask:  (batch_size, 1, sequence_length, sequence_length)
        """

        x = self.source_embedding(encoder_input)
        x = self.positional_encoding(x)
        x = self.encoder(x, source_mask)

        encoder_output = x
        x = self.source_embedding(decoder_input)
        x = self.positional_encoding(x)
        x = self.decoder(x, encoder_output, source_mask, target_mask)

        x = self.projection_layer(x)
        return x
