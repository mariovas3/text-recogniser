from typing import Sequence

import torchmetrics


class CharacterErrorRate(torchmetrics.CharErrorRate):
    """
    Character error rate metric, allowing for tokens to be ignored.

    Direct copy from:
    https://github.com/the-full-stack/fsdl-text-recognizer-2022-labs/blob/main/lab03/text_recognizer/lit_models/metrics.py
    """

    def __init__(self, ignore_tokens: Sequence[int], *args):
        super().__init__(*args)
        self.ignore_tokens = set(ignore_tokens)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):  # type: ignore
        preds_l = [
            [t for t in pred if t not in self.ignore_tokens]
            for pred in preds.tolist()
        ]
        targets_l = [
            [t for t in target if t not in self.ignore_tokens]
            for target in targets.tolist()
        ]
        super().update(preds_l, targets_l)
