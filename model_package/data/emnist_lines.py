import random
from collections import defaultdict
from typing import Tuple

import h5py
import numpy as np
import torch
import torchvision.transforms as T

import model_package.metadata.emnist_lines as metadata
from model_package.data.emnist import EMNIST
from model_package.data.lit_datamodule import BaseDataModule, mock_lit_dataset
from model_package.data.sentence_generator import SentenceGenerator
from model_package.data.utils import (
    SupervisedDataset,
    convert_strings_to_labels,
)

DEFAULT_MAX_LENGTH = 32
DEFAULT_MIN_OVERLAP = 0
DEFAULT_MAX_OVERLAP = 0.33

NUM_TRAIN = 10000
NUM_VAL = 2000
NUM_TEST = 2000


class EMNISTLines(BaseDataModule):
    def __init__(
        self,
        max_length=DEFAULT_MAX_LENGTH,
        min_overlap=DEFAULT_MIN_OVERLAP,
        max_overlap=DEFAULT_MAX_OVERLAP,
        num_train=NUM_TRAIN,
        num_val=NUM_VAL,
        num_test=NUM_TEST,
        with_start_and_end_tokens=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.with_start_and_end_tokens = with_start_and_end_tokens

        self.idx_to_char = metadata.MAPPING
        self.char_to_idx = {c: i for i, c in enumerate(metadata.MAPPING)}
        self.emnist_lit = EMNIST()
        self.emnist_lit.prepare_data()
        self.transform = T.ToTensor()
        self.input_dims = (
            1,
            metadata.CHAR_HEIGHT,
            metadata.CHAR_WIDTH * self.max_length,
        )
        self.output_dims = (self.max_length,)

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument(
            "--max_length",
            type=int,
            default=DEFAULT_MAX_LENGTH,
            help=f"Max num chars per line. Default is {DEFAULT_MAX_LENGTH}.",
        )
        parser.add_argument(
            "--min_overlap",
            type=float,
            default=DEFAULT_MIN_OVERLAP,
            help=f"Min overlap between chars in between 0 and 1. Default is {DEFAULT_MIN_OVERLAP}",
        )
        parser.add_argument(
            "--max_overlap",
            type=float,
            default=DEFAULT_MAX_OVERLAP,
            help=f"Max overlap between chars in between 0 and 1. Default is {DEFAULT_MAX_OVERLAP}",
        )
        parser.add_argument(
            "--with_start_and_end_tokens", action="store_true", default=True
        )
        return parser

    @property
    def data_filenames(self) -> Tuple[str]:
        base = (
            f"ml_{self.max_length}_o{self.min_overlap:f}_{self.max_overlap:f}"
        )
        suffs = (
            f"_ntrain{self.num_train}_start_end_{self.with_start_and_end_tokens}.h5",
            f"_nval{self.num_val}_start_end_{self.with_start_and_end_tokens}.h5",
            f"_ntest{self.num_test}_start_end_{self.with_start_and_end_tokens}.h5",
        )
        return [metadata.PROCESSED_DATA_DIR / (base + suff) for suff in suffs]

    def prepare_data(self) -> None:
        if not self.data_filenames[0].exists():
            self._generate_data("fit")
        if not self.data_filenames[-1].exists():
            self._generate_data("test")

    def _generate_data(self, split):
        print(f"EMNISTLinesDataset generating data for {split}...")
        assert split in ("fit", "test")
        # save 2 spaces for <START> and <END> tokens;
        sentence_generator = SentenceGenerator(max_length=self.max_length - 2)

        emnist = self.emnist_lit
        emnist.setup(split)

        if split == "fit":
            imgs_by_char = get_images_by_char(
                emnist.x_train_and_val,
                emnist.y_train_and_val,
                self.idx_to_char,
            )
            num_train, num_val = self.num_train, self.num_val
            num = num_train + num_val
        else:
            imgs_by_char = get_images_by_char(
                emnist.x_test, emnist.y_test, self.idx_to_char
            )
            num = self.num_test

        metadata.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        train_file, val_file, test_file = self.data_filenames

        x, y = create_emnistlines(
            num,
            imgs_by_char,
            sentence_generator,
            min_overlap=self.min_overlap,
            max_overlap=self.max_overlap,
            dims=self.input_dims,
        )
        y = convert_strings_to_labels(
            y,
            self.char_to_idx,
            length=self.max_length,
            with_start_and_end_tokens=self.with_start_and_end_tokens,
        )

        if split == "fit":
            x_train, x_val = x[:num_train], x[num_train:]
            y_train, y_val = y[:num_train], y[num_train:]

            with h5py.File(train_file, "w") as f:
                f.create_dataset(
                    "x_train", data=x_train, dtype="u1", compression="lzf"
                )
                f.create_dataset(
                    "y_train", data=y_train, dtype="u1", compression="lzf"
                )

            with h5py.File(val_file, "w") as f:
                f.create_dataset(
                    "x_val", data=x_val, dtype="u1", compression="lzf"
                )
                f.create_dataset(
                    "y_val", data=y_val, dtype="u1", compression="lzf"
                )
        else:
            with h5py.File(test_file, "w") as f:
                f.create_dataset(
                    "x_test", data=x, dtype="u1", compression="lzf"
                )
                f.create_dataset(
                    "y_test", data=y, dtype="u1", compression="lzf"
                )

    def setup(self, stage):
        assert stage in ("fit", "test")
        train_file, val_file, test_file = self.data_filenames
        print("EMNISTLines loading from HDFS...")
        if stage == "fit":
            with h5py.File(train_file, "r") as f:
                x_train = f["x_train"][:]
                # cast to int important;
                y_train = f["y_train"][:].astype(int)

            with h5py.File(val_file, "r") as f:
                x_val = f["x_val"][:]
                # cast to int important;
                y_val = f["y_val"][:].astype(int)

            self.train_dataset = SupervisedDataset(
                x_train, y_train, inputs_transform=self.transform
            )
            self.val_dataset = SupervisedDataset(
                x_val, y_val, inputs_transform=self.transform
            )
        else:
            with h5py.File(test_file, "r") as f:
                x_test = f["x_test"][:]
                y_test = f["y_test"][:].astype(int)

            self.test_dataset = SupervisedDataset(
                x_test, y_test, inputs_transform=self.transform
            )

    def __repr__(self) -> str:
        return (
            "EMNIST Lines Dataset\n"
            f"Min overlap: {self.min_overlap}\n"
            f"Max overlap: {self.max_overlap}\n"
            f"Num unique tokens: {len(self.idx_to_char)}\n"
            f"Dims: {self.input_dims}\n"
            f"Output dims: {self.output_dims}\n"
        )


def get_images_by_char(x, y, idx_to_char):
    """Group images by character."""
    grouping = defaultdict(list)
    for img, label_idx in zip(x, y):
        grouping[idx_to_char[label_idx]].append(img)
    return grouping


def create_emnistlines(
    N, imgs_by_char, sentence_generator, min_overlap, max_overlap, dims
):
    """Get tensor (N, dims[1], dims[2]) of images and list of N strings as labels."""
    # num_samples x Height x CharWidth * max_len
    images = torch.zeros((N, dims[1], dims[2]))
    labels = []
    for n in range(N):
        # returns sentence string;
        label = sentence_generator.generate()
        # store image of label;
        images[n] = make_image_from_string(
            label, imgs_by_char, min_overlap, max_overlap, dims[-1]
        )
        labels.append(label)
    return images, labels


def make_image_from_string(
    string_label, imgs_by_char, min_overlap, max_overlap, max_width
):
    """Concat images of chars in string_label to one image."""
    overlap = random.uniform(min_overlap, max_overlap)
    sampled_images = get_image_for_string_label(string_label, imgs_by_char)
    # see shape of first image;
    H, W = sampled_images[0].shape
    # see how close to each other the characters will be;
    next_overlap_width = W - int(overlap * W)
    # init with black image;
    concat_image = torch.zeros((H, max_width), dtype=torch.uint8)
    x = 0
    # start "writing"
    for image in sampled_images:
        concat_image[:, x : (x + W)] += image
        x += next_overlap_width
    # images pixels are still in [0, 255], so not sure why the minimum is needed;
    return torch.minimum(torch.tensor([255]), concat_image)


def get_image_for_string_label(
    string_label,
    imgs_by_char,
    char_shape=(metadata.CHAR_HEIGHT, metadata.CHAR_WIDTH),
):
    """Return list of images - np.ndarray for existing and torch.Tensor for missing chars."""
    zero_image = torch.zeros(char_shape, dtype=torch.uint8)
    sampled_imgs_by_char = {}
    for char in string_label:
        # assuming perfect style of writing for each char
        # already written by user;
        if char in sampled_imgs_by_char:
            continue
        if imgs_by_char[char]:
            # if imgs_by_char has np.ndarray images,
            # we could return a list of a mix of np.ndarray and torch.Tensor
            # objects;
            sample = random.choice(imgs_by_char[char])
        else:
            sample = zero_image
        sampled_imgs_by_char[char] = sample.reshape(*char_shape)
    return [sampled_imgs_by_char[char] for char in string_label]


if __name__ == "__main__":
    import random

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    def test_plot(data_batch, batch_size, min_idx=0):
        to_show = min(10, batch_size)
        for i in range(to_show):
            x, y = next(data_batch)
            plt.subplot(10, 1, i + 1)
            plt.imshow(x.squeeze().numpy(), cmap="gray")
            label = "".join(
                [metadata.MAPPING[idx.item()] for idx in y if idx >= min_idx]
            )
            plt.title(label)
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    emnist_lines_dataset, args = mock_lit_dataset(EMNISTLines)
    emnist_lines_dataset.prepare_data()
    emnist_lines_dataset.setup(args["stage"])
    print(emnist_lines_dataset)
    if args["stage"] == "fit":
        dl = iter(emnist_lines_dataset.train_dataloader())
    else:
        dl = emnist_lines_dataset.test_dataloader()
    x, y = next(iter(dl))
    print(f"x.shape: {x.shape}, y.shape: {y.shape}")
    print(
        f"x dtype, min, mean, max, std: {(x.dtype, x.min(), x.mean(), x.max(), x.std())}"
    )
    print(f"y dtype, min, max: {(y.dtype, y.min(), y.max())}")
    print(f"batch_size: {args['batch_size']}")
    test_plot(zip(x, y), args["batch_size"], min_idx=4)
