import random

from torch.utils.data import Dataset

import model_package.metadata.iam_synthetic_paragraphs as metadata
from model_package.data import utils
from model_package.data.iam import IAM
from model_package.data.iam_lines import (
    generate_line_crops_and_labels,
    load_processed_line_crops,
    load_processed_line_labels,
    save_images_and_labels,
)
from model_package.data.iam_paragraphs import IAMParagraphs


class IAMSyntheticParagraphs(IAMParagraphs):
    def __init__(self, dataset_len=metadata.DATASET_LEN, **kwargs):
        super().__init__(**kwargs)
        self.line_crops = None
        self.line_labels = None
        self.dataset_len = dataset_len

    def prepare_data(self) -> None:
        if metadata.PROCESSED_DATA_DIR.exists():
            print(
                f"PROCESSED DIR already exists: {metadata.PROCESSED_DATA_DIR}"
            )
            return

        iam = IAM()
        iam.prepare_data()

        # synthetic dataset only for training phase;
        print(f"Preparing IAMSyntheticParagraphs...")
        crops, labels = generate_line_crops_and_labels(iam, "train")
        save_images_and_labels(
            crops, labels, "train", metadata.PROCESSED_DATA_DIR
        )

    def setup(self, stage) -> None:
        assert stage == "fit"
        print(f"Setup Train Synthetic Paragraphs...")
        self._load_processed_crops_and_labels()
        self.train_dataset = IAMSyntheticParagraphsDataset(
            self.line_crops,
            self.line_labels,
            self.dataset_len,
            self.char_to_idx,
            self.input_dims,
            self.output_dims,
            self.trainval_transform,
        )

    def _load_processed_crops_and_labels(self) -> None:
        if self.line_crops is None:
            self.line_crops = load_processed_line_crops(
                "train", metadata.PROCESSED_DATA_DIR
            )
        if self.line_labels is None:
            self.line_labels = load_processed_line_labels(
                "train", metadata.PROCESSED_DATA_DIR
            )

    def __repr__(self):
        ans = (
            "IAMSyntheticParagraphs Dataset\n"
            f"Num classes: {len(self.idx_to_char)}\n"
            f"Input dims: {self.input_dims}\n"
            f"Output dims: {self.output_dims}\n"
        )
        return ans


class IAMSyntheticParagraphsDataset(Dataset):
    def __init__(
        self,
        line_crops,
        line_labels,
        dataset_len,
        char_to_idx,
        input_dims,
        output_dims,
        transform,
    ):
        super().__init__()
        assert len(line_crops) == len(line_labels)
        print(f"len_line_crops: {len(line_labels)}")
        self.line_crops = line_crops
        self.line_labels = line_labels
        self.ids = list(range(len(self.line_labels)))
        self.dataset_len = dataset_len
        self.char_to_idx = char_to_idx
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.transform = transform
        self.min_num_lines, self.max_num_lines = 1, 15

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        # this whole sampling thing should be precomputed somewhere;
        num_lines = random.randint(self.min_num_lines, self.max_num_lines)
        indices = random.sample(self.ids, k=num_lines)

        while True:
            datum = utils.hstack_line_crops(
                [self.line_crops[i] for i in indices]
            )
            labels = "\n".join([self.line_labels[i] for i in indices])
            if (
                len(labels) <= self.output_dims[0] - 2
                and datum.height <= self.input_dims[1]
                and datum.width <= self.input_dims[2]
            ):
                break
            indices = indices[:-1]
        if self.transform is not None:
            datum = self.transform(datum)
        length = self.output_dims[0]
        target = utils.convert_strings_to_labels(
            [labels],
            self.char_to_idx,
            length=length,
            with_start_and_end_tokens=True,
        )[0]
        return datum, target


if __name__ == "__main__":
    import random

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from model_package.data.lit_datamodule import mock_lit_dataset

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    def test_plot(data_batch, batch_size, idx_to_char):
        fig = plt.figure(figsize=(12, 12))
        to_show = min(9, batch_size)
        for i in range(to_show):
            x, y = next(data_batch)
            plt.subplot(3, 3, i + 1)
            plt.imshow(x.squeeze().numpy(), cmap="gray")
            num_lines = sum(idx_to_char[idx.item()] == "\n" for idx in y) + 1
            plt.title(f"{num_lines =}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    iam_par_dataset, args = mock_lit_dataset(IAMSyntheticParagraphs)
    iam_par_dataset.prepare_data()
    iam_par_dataset.setup(args["stage"])
    print(iam_par_dataset)
    if args["stage"] == "fit":
        dl = iter(iam_par_dataset.train_dataloader())
    else:
        dl = iter(iam_par_dataset.test_dataloader())
    x, y = next(dl)
    # for _ in range(9):
    #     text = ''.join(iam_par_dataset.idx_to_char[idx.item()] for idx in y[_])
    #     print(text, end='\n\n')
    print(f"x.shape: {x.shape}, y.shape: {y.shape}")
    print(
        f"x dtype, min, mean, max, std: {(x.dtype, x.min(), x.mean(), x.max(), x.std())}"
    )
    print(f"y dtype, min, max: {(y.dtype, y.min(), y.max())}")
    print(f"batch_size: {len(y)}")
    test_plot(zip(x, y), len(y), idx_to_char=iam_par_dataset.idx_to_char)
