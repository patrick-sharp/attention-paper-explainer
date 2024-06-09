import torch
from torchmetrics.text import BLEUScore
from torch.utils.data import DataLoader

import components
import dataset
from translation import translate_single


def test_model(components):
    """Compute the BLEU metric for the test set"""
    model = components.model
    test_batched = components.test_batched
    test_tokenized = components.test_tokenized

    test_dataloader = DataLoader(test_batched, batch_size=None, shuffle=True)

    model.eval()

    expected_text = []
    predicted_text = []

    config = components.config
    tokenizer = components.tokenizer
    pad_token = config.pad_token
    pad_token_id = tokenizer.token_to_id(pad_token)

    for batch in test_dataloader:
        # batch size always 1 here
        source_text = batch["source_text"][0]
        target_text = batch["target_text"][0]

        # encoder_input = torch.tensor(batch["de_tok"], dtype=torch.int32).unsqueeze(0)
        # source_mask = dataset.create_source_mask(encoder_input, pad_token_id)

        # bleu expects multiple expected translations, so make a singleton list.
        # the wmt14 dataset only has one english translation per german sentence
        expected_text.append([target_text])
        # translation = translate_tensor(components, encoder_input, source_mask)
        _, translation = translate_single(components, source_text)
        predicted_text.append(translation)

    # default n gram is 4, so won't consider sentences 3 words or less
    metric = BLEUScore()
    bleu = metric.update(predicted_text, expected_text)
    bleu = metric.compute()

    return bleu.item()
