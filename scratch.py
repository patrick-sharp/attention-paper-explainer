# scratchpad for experimenting with different python functions

from torchmetrics.text import BLEUScore


def bleu():
    predicted_text = ["this is so so great"]
    expected_text = [["this is so so great", "this is fine"]]
    metric = BLEUScore()
    print(predicted_text)
    print(expected_text)
    bleu = metric.update(predicted_text, expected_text)
    bleu = metric.compute()
    print(bleu.item())
