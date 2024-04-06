import torch
import torch.nn as nn
import torch.nn.functional as functional

import math

from config import DEFAULT_CONFIG


# Note: no dropout. Dropout is applied to the sum of the embedding and
# positional encoding
class ScaledEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.embedding = nn.Embedding(config.tokenizer_vocab_size, config.d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.p_dropout)

        # (sequence_length, d_model)
        pos_enc = torch.zeros(config.sequence_length, config.d_model)

        # (sequence_length)
        position = torch.arange(0, config.sequence_length, dtype=torch.float)

        # (sequence_length, 1)
        position = position.unsqueeze(1)

        # (d_model / 2)
        evens = torch.arange(0, config.d_model, 2).float()

        # (d_model / 2)
        exponent = evens / config.d_model

        # (d_model / 2)
        denominator = torch.pow(torch.ones(config.d_model // 2) * 10000.0, exponent)

        # sine even indices
        pos_enc[:, 0::2] = torch.sin(position / denominator)

        # cosine odd indices
        pos_enc[:, 1::2] = torch.cos(position / denominator)

        # (1, sequence_length, d_model)
        pos_enc = pos_enc.unsqueeze(0)

        # Register the positional encoding as a buffer
        self.register_buffer("pos_enc", pos_enc)

    def forward(self, x):
        # broadcasts addition over all sequences in batch
        # (batch_size, sequence_length, d_model)
        x = x + self.pe

        return self.dropout(x)


class SelfAttention(nn.Module):
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
        self.w_v = nn.Linear(d_model, d_value, bias=bias)
        self.w_v_dropout = nn.Dropout(p_dropout)

        # combines all heads
        # (num_heads * d_value, d_model)
        self.w_o = nn.Linear(num_heads * d_value, d_model, bias=bias)
        
        # in Karpathy's tutorial, he uses two dropouts.
        # The paper seems to only imply one (at the end of each sub-layer).
        # not sure why there's a difference, but there you go.
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        batch_size = self.config.batch_size
        sequence_length = self.config.sequence_length
        d_model = self.config.d_model
        d_key = self.config.d_key
        d_value = self.config.d_value
        num_heads = self.config.num_heads
        bias = self.config.bias
        p_dropout = self.config.p_dropout

        # calculate the queries, keys, and values for all heads at once
        # INTERMISSION: learn about the rules for batched matrix multiplies
        #     in torch.matmul

        # (batch_size, sequence_length, num_heads * d_key)
        q = self.w_q(x)

        # (batch_size, sequence_length, num_heads * d_key)
        k = self.w_k(x)

        # (batch_size, sequence_length, num_heads * d_value)
        v = self.w_v(x)

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
        # softmax to get probabilities
        x = functional.softmax(attention, dim=-1)
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
        x = x.contiguous().view(batch_size, sequence_length, num_heads * d_model)

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

class EncoderBlock(nn.Module):
    def __init__(self, config):
        pass


class Encoder(nn.Module):
    def __init__(self, config):
        pass


class DecoderBlock(nn.Module):
    def __init__(self, config):
        pass


class Decoder(nn.Module):
    def __init__(self, config):
        pass


class Transformer(nn.Module):
    def __init__(self, config=DEFAULT_CONFIG):
        super().__init__()

        self.source_embeddings = ScaledEmbeddings(config)
        self.source_pos_enc = PositionalEncoding(config)
        self.encoder = Encoder(config)

        # self.target_embeddings = ScaledEmbeddings(config)
        # self.decoder = Decoder(config)

    def forward(self, x):
        """training forward pass, not translation"""
