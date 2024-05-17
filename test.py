import torch
from torchmetrics.text import BLEUScore
from torch.utils.data import DataLoader

import components
import dataset
from translate import translate_tensor


def test_model(components):
    """Compute the BLEU metric for the test set"""
    model = components.model
    test_batched = components.test_batched
    test_tokenized = components.test_tokenized

    # test_dataloader = DataLoader(test_batched, batch_size=None, shuffle=True)
    # test_dataloader = DataLoader(test_tokenized, batch_size=None, shuffle=True)
    test_dataloader = DataLoader(
        components.train_tokenized, batch_size=None, shuffle=True
    )

    model.eval()

    expected_text = []
    predicted_text = []

    config = components.config
    tokenizer = components.tokenizer
    pad_token = config.pad_token
    pad_token_id = tokenizer.token_to_id(pad_token)

    for batch in test_dataloader:
        source_text = batch["de"]
        target_text = batch["en"]

        encoder_input = torch.tensor(batch["de_tok"], dtype=torch.int32).unsqueeze(0)
        source_mask = dataset.create_source_mask(encoder_input, pad_token_id)

        expected_text.append([target_text])
        translation = translate_tensor(components, encoder_input, source_mask)
        predicted_text.append(translation)

    metric = BLEUScore()
    print(predicted_text)
    print(expected_text)
    bleu = metric.update(predicted_text, expected_text)
    bleu = metric.compute()

    return bleu.item()
