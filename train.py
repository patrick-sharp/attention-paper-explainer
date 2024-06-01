from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset
import model
from configuration import DEFAULT_CONFIG
from keyboard_interrupt import DelayedKeyboardInterrupt
from translate import translate_sentence


def init_optimizer(config, model):
    adam_epsilon = config.adam_epsilon
    adam_learning_rate = config.adam_learning_rate
    adam_beta_1 = config.adam_beta_1
    adam_beta_2 = config.adam_beta_2

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=adam_learning_rate,
        eps=adam_epsilon,
        betas=(adam_beta_1, adam_beta_2),
    )

    return optimizer


def benchmark_translations(components, examples):
    epoch_translations = []
    for example in examples:
        example_copy = dict(example)
        example_copy["translation"] = translate_sentence(components, example["source"])
        epoch_translations.append(example_copy)
    return epoch_translations


def train_model(components):
    config = components.config

    pad_token = config.pad_token
    label_smoothing_epsilon = config.label_smoothing_epsilon

    train_steps = config.train_steps
    vocab_size = components.tokenizer.get_vocab_size()

    tokenizer = components.tokenizer
    train_batched = components.train_batched
    examples = train_batched.examples

    MODEL_TRAIN_STATE = components.types.MODEL_TRAIN_STATE
    if not components.present[MODEL_TRAIN_STATE]:
        components.create(MODEL_TRAIN_STATE)
    model = components.model
    optimizer = components.optimizer
    losses = components.losses
    translations = components.translations

    # batch_size=None disables automatic addition of batch dimension
    train_dataloader = DataLoader(train_batched, batch_size=None, shuffle=True)

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer.token_to_id(pad_token),
        label_smoothing=label_smoothing_epsilon,
    )

    if components.epoch == 0:
        # it's funny to see what a completely
        # untrained model will predict, so make that
        # the first group of example translations
        epoch_translations = benchmark_translations(components, examples)
        translations.append(epoch_translations)

    for epoch in range(components.epoch, train_steps):
        model.train()  # switch back to training after running test benchmark
        batch_tqdm = tqdm(train_dataloader, desc=f"Epoch {epoch:04d}")
        for batch in batch_tqdm:
            encoder_input = batch["encoder_input"]
            decoder_input = batch["decoder_input"]
            source_mask = batch["source_mask"]
            target_mask = batch["target_mask"]
            label = batch["label"]

            # (batch_size, seq_len, vocab_size)
            output = model(encoder_input, decoder_input, source_mask, target_mask)

            # flatten out predictions and labels because the loss function
            # for class predictions expects 2D predictions and 1D labels

            # (batch_size * seq_len, vocab_size)
            loss_input = output.view(-1, vocab_size)

            # (batch_size * seq_len)
            # cross entropy expects a long, but loss_target defaults to int
            loss_target = label.view(-1).long()

            # compute the loss and the gradients of the model parameters
            loss = loss_fn(loss_input, loss_target)
            # print the loss after each batch
            batch_tqdm.set_postfix({"loss": f"{loss.item():.3f}"})
            # send the gradients to the model parameters
            loss.backward()

            # keep track of the loss for this batch
            losses.append(loss.item())

            # update model parameters
            optimizer.step()
            # set gradients to None so they don't affect the next batch
            optimizer.zero_grad(set_to_none=True)

        # track the progress of the model on example sentences after each epoch
        epoch_translations = benchmark_translations(components, examples)
        translations.append(epoch_translations)

        # Save the train state at the end of every epoch
        with DelayedKeyboardInterrupt():
            components.save(
                MODEL_TRAIN_STATE,
                {
                    "epoch": epoch,
                    "losses": losses,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "translations": translations,
                },
            )


if __name__ == "__main__":
    cmp = components.Components(DEFAULT_CONFIG)
    train_transformer(cmp)
