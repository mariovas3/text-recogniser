from typing import Sequence

import torch
from torchmetrics.text import CharErrorRate


class MyCharErrorRate(CharErrorRate):
    """
    Character error rate metric, allowing for tokens to be ignored.

    Calculation from lightning docs is:

    Rate = (S + D + I) / (S + D + C)

    Basically the numerator is the edit_distance(pred, target)
    and you win if C > I. Lower is better - ideally 0; can be
    greater than 1, however.

    Where:
        S - num substitutions
        D - num deletions
        I - num insertions
        C - num correct chars

    Direct copy from:
    https://github.com/the-full-stack/fsdl-text-recognizer-2022-labs/blob/main/lab03/text_recognizer/lit_models/metrics.py

    The errros are accumulated:
    https://github.com/Lightning-AI/torchmetrics/blob/v1.3.1/src/torchmetrics/text/cer.py#L28-L138

    Technically preds and targets are annotated to be str or list[str]
    but based on the code in the docs, list[seq] also works;
    https://github.com/Lightning-AI/torchmetrics/blob/v1.3.1/src/torchmetrics/functional/text/cer.py#L47
    """

    def __init__(self, ignore_tokens: Sequence[int], **kwargs):
        super().__init__(**kwargs)
        self.ignore_tokens = set(ignore_tokens)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds_l = [
            [t for t in pred if t not in self.ignore_tokens]
            for pred in preds.tolist()
        ]
        targets_l = [
            [t for t in target if t not in self.ignore_tokens]
            for target in targets.tolist()
        ]
        super().update(preds_l, targets_l)


if __name__ == "__main__":
    X = torch.tensor(
        [
            [0, 2, 2, 3, 3, 1],  # cum_error=0 single cer 0
            [0, 2, 1, 1, 1, 1],  # cum_error=3 D=3 C=1 single cer .75
            [0, 2, 2, 4, 4, 1],  # cum_error=5 S=2 C=2 single cer .5
        ]
    )
    Y = torch.tensor(
        [
            [0, 2, 2, 3, 3, 1],  # cum_len=4
            [0, 2, 2, 3, 3, 1],  # cum_len=8
            [0, 2, 2, 3, 3, 1],  # cum_len=12
        ]
    )
    cum_errors = [0, 3, 5]
    cum_lens = [4, 8, 12]
    cer_fn = MyCharErrorRate([0, 1])
    cer1 = cer_fn(X, Y)  # first forward pass eval;
    assert cer1 == cum_errors[-1] / cum_lens[-1]
    cer2 = cer_fn(X[:2], Y[:2])  # second forward pass eva;
    assert cer2 == cum_errors[-2] / cum_lens[-2]
    assert cer_fn.errors == 5 + 3  # accumulates
    assert cer_fn.total == 12 + 8  # accumulates
