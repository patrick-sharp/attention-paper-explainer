import matplotlib.pyplot as plt
import torch

from global_state import config, cmp
import masking
import translation


# plot a color mesh of the positional encodings
def positional_encodings():
    """This code is from Jay Alammar"""

    if not cmp.present[MODEL_TRAIN_STATE]:
        cmp.create(MODEL_TRAIN_STATE)

    tokens = 10

    # (10, d_model)
    pos_encoding = cmp.model.positional_encoding.positional_encodings[
        0, 0:tokens, :
    ].detach()

    plt.figure(figsize=(12, 8))
    plt.pcolormesh(pos_encoding, cmap="viridis")
    plt.xlabel("Embedding Dimensions")
    plt.xlim((0, cmp.config.d_model))
    plt.ylim((tokens, 0))
    plt.ylabel("Token Position")
    plt.colorbar()
    plt.show()


# plot a graph of the current loss
def loss():
    plt.plot(cmp.losses)
    plt.ylabel("Loss")
    plt.xlabel("Batch")
    ax = plt.gca()
    ax.set_ylim([0, cmp.losses[0] + 1])
    plt.show()


def encoder_attention(layer=0, head=0, sentence=translation.en_1):
    tokenizer = cmp.tokenizer
    model = cmp.model
    pad_token_id = cmp.pad_token_id

    model.eval()
    token_ids = tokenizer.encode(sentence).ids
    tokens = [tokenizer.id_to_token(token_id) for token_id in token_ids]
    encoder_input = torch.tensor(token_ids).unsqueeze(0)
    source_mask = masking.create_source_mask(encoder_input, pad_token_id)

    instrumentation = {
        "layer": layer,
        "head": head,
    }
    # for side effects on instrumentation
    model.encode(encoder_input, source_mask, instrumentation)
    attention = instrumentation["attention"].transpose(0, 1).flip([1])

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(attention)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(range(len(tokens)), labels=tokens)
    ax.set_yticks(range(len(tokens)), labels=reversed(tokens))
    plt.xlabel("Query")
    plt.ylabel("Key")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(attention.shape[0]):
        for j in range(attention.shape[1]):
            text = ax.text(
                j,
                i,
                "{:.2f}".format(attention[i, j].item()),
                ha="center",
                va="center",
                color="w",
            )

    ax.set_title("Attention scores")
    fig.tight_layout()
    plt.show()
    return attention


def embeddings():
    model = cmp.model

    source = (
        model.source_embedding.embedding._parameters["weight"].detach().transpose(0, 1)
    )
    target = (
        model.target_embedding.embedding._parameters["weight"].detach().transpose(0, 1)
    )

    import numpy as np

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    plt.subplot(211)
    im = ax1.imshow(source)
    plt.xlabel("Token id")
    plt.ylabel("Embedding vector dimension")
    ax1.set_title("Source embeddings")

    plt.subplot(212)
    im = ax2.imshow(target)
    plt.xlabel("Token id")
    plt.ylabel("Embedding vector dimension")
    ax2.set_title("Target embeddings")

    fig.tight_layout()
    plt.show()
