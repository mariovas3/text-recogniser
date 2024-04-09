from argparse import ArgumentParser

from torch.utils.data import ConcatDataset

from model_package.data.iam_paragraphs import IAMParagraphs
from model_package.data.iam_synthetic_paragraphs import IAMSyntheticParagraphs
from model_package.data.lit_datamodule import BaseDataModule
from model_package.metadata.iam_synthetic_paragraphs import DATASET_LEN


class IAMOgAndSynthParagraphs(BaseDataModule):
    def __init__(
        self,
        augment=False,
        dataset_len=DATASET_LEN,
        use_precomputed=False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.augment = augment
        self.iam_paragraphs = IAMParagraphs(augment=augment, **kwargs)
        self.iam_synth_paragraphs = IAMSyntheticParagraphs(
            dataset_len=dataset_len,
            augment=augment,
            use_precomputed=use_precomputed,
            **kwargs,
        )
        self.input_dims = self.iam_paragraphs.input_dims
        self.output_dims = self.iam_paragraphs.output_dims
        self.idx_to_char = self.iam_paragraphs.idx_to_char
        self.char_to_idx = self.iam_paragraphs.char_to_idx

    # only for quick testing purposes; all parsing is
    # handled by LightningCLI;
    @staticmethod
    def add_to_argparse(parser: ArgumentParser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument(
            "--augment",
            action="store_true",
            help="Whether to augment train and val images.",
        )
        parser.add_argument(
            "--dataset_len",
            type=int,
            default=DATASET_LEN,
            help=f"Length of Synthetic Paragraphs Dataset, default is {DATASET_LEN}",
        )
        parser.add_argument(
            "--use_precomputed",
            action="store_true",
            help="Whether to check if precomputed synth paragraphs available.",
        )
        return parser

    def prepare_data(self) -> None:
        self.iam_paragraphs.prepare_data()
        self.iam_synth_paragraphs.prepare_data()

    def setup(self, stage):
        assert stage in ("fit", "test")
        self.iam_paragraphs.setup(stage)
        if stage == "fit":
            self.iam_synth_paragraphs.setup(stage)
            self.train_dataset = ConcatDataset(
                (
                    self.iam_paragraphs.train_dataset,
                    self.iam_synth_paragraphs.train_dataset,
                )
            )
            self.val_dataset = self.iam_paragraphs.val_dataset
        else:
            self.test_dataset = self.iam_paragraphs.test_dataset

    def __repr__(self):
        ans = (
            "IAM Original and Synthetic Paragraphs Dataset\n"
            f"Num classes: {len(self.idx_to_char)}\n"
            f"Input Dims: {self.input_dims}\n"
            f"Output Dims: {self.output_dims}\n"
        )
        return ans


if __name__ == "__main__":
    from model_package.data.testing_utils import (
        get_info_litdata,
        test_plot_para,
    )

    x, y, idx_to_char = get_info_litdata(IAMOgAndSynthParagraphs)
    test_plot_para(zip(x, y), len(y), idx_to_char=idx_to_char)
