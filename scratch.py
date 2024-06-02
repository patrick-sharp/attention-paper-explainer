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


def batches(cmp):
    total_tokens = 0
    total_tokens_wasted = 0
    for b in cmp.train_batched.batch_bounds:
        blen = b["end"] - b["start"]
        batch_tokens = b["max_len"] * blen * 2
        oop = blen * (b["md"] + b["me"])
        tokens_wasted = batch_tokens - oop
        total_tokens_wasted += tokens_wasted

        total_tokens += batch_tokens
    # generally tokens wasted without different lengths is about 5%
    return total_tokens_wasted, total_tokens, total_tokens_wasted / total_tokens
