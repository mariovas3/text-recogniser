import json
from functools import partial, reduce
from itertools import chain
from pathlib import Path

import torchvision.transforms as T
from PIL import Image

import model_package.data.utils as utils
import model_package.metadata.iam_lines as metadata
from model_package.data.iam import IAM
from model_package.data.lit_datamodule import BaseDataModule, mock_lit_dataset

resize_crop = partial(
    utils.resize_crop,
    new_width=metadata.IMAGE_WIDTH,
    new_height=metadata.IMAGE_HEIGHT,
)


class IAMLines(BaseDataModule):
    """Lines of text extracted from the IAM handwriting database."""

    def __init__(self, augment=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.augment = augment
        self.idx_to_char = metadata.MAPPING
        self.char_to_idx = {c: i for i, c in enumerate(metadata.MAPPING)}
        self.input_dims = metadata.DIMS
        self.output_dims = metadata.OUTPUT_DIMS
        self.transform = IAMLinesTransforms()
        self.trainval_transform = IAMLinesTransforms(augment=augment)

    def prepare_data(self) -> None:
        if metadata.PROCESSED_DATA_DIR.exists():
            print(
                f"PROCESSED DIR already exists: {metadata.PROCESSED_DATA_DIR}"
            )
            return

        print("Preparing IAMLines...")
        metadata.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        iam = IAM()
        iam.prepare_data()
        print("Getting Line Crops and Line Labels...")
        crops_train, labels_train = generate_line_crops_and_labels(
            iam, "train"
        )
        crops_val, labels_val = generate_line_crops_and_labels(iam, "val")
        crops_test, labels_test = generate_line_crops_and_labels(iam, "test")

        # save max width to height ratio (aspect ratio) over each crop;
        max_aspect_ratio = reduce(
            lambda val, ob: max(val, ob.size[0] / ob.size[1]),
            chain(crops_train, crops_val, crops_test),
            0,  # init with 0;
        )
        with open(
            metadata.PROCESSED_DATA_DIR / "_max_aspect_ratio.txt", "w"
        ) as file:
            file.write(str(max_aspect_ratio))

        # save images and labels;
        print("Saving crops and labels...")
        save_images_and_labels(
            crops_train, labels_train, "train", metadata.PROCESSED_DATA_DIR
        )
        save_images_and_labels(
            crops_val, labels_val, "val", metadata.PROCESSED_DATA_DIR
        )
        save_images_and_labels(
            crops_test, labels_test, "test", metadata.PROCESSED_DATA_DIR
        )

    def setup(self, stage) -> None:
        assert stage in ("fit", "test")
        with open(
            metadata.PROCESSED_DATA_DIR / "_max_aspect_ratio.txt"
        ) as file:
            max_aspect_ratio = float(file.read())
            max_crop_width = int(metadata.IMAGE_HEIGHT * max_aspect_ratio)
            assert (
                max_crop_width <= metadata.IMAGE_WIDTH
            ), f"max_crop_width: {max_crop_width}, max_allowed: {metadata.IMAGE_WIDTH}"

        if stage == "fit":
            x_train, labels_train = load_processed_crops_and_labels(
                "train", metadata.PROCESSED_DATA_DIR
            )
            # output dims is set to 89 since empirically longest handwritten
            # line has 87 chars, we add 2 for the start and end tokens;
            # if some label has more than 87 chars, convert_strings_to_labels
            # will raise an IndexError;
            y_train = utils.convert_strings_to_labels(
                labels_train,
                self.char_to_idx,
                length=self.output_dims[0],
                with_start_and_end_tokens=True,
            )
            self.train_dataset = utils.SupervisedDataset(
                x_train, y_train, inputs_transform=self.trainval_transform
            )

            x_val, labels_val = load_processed_crops_and_labels(
                "val", metadata.PROCESSED_DATA_DIR
            )
            y_val = utils.convert_strings_to_labels(
                labels_val,
                self.char_to_idx,
                length=self.output_dims[0],
                with_start_and_end_tokens=True,
            )
            self.val_dataset = utils.SupervisedDataset(
                x_val, y_val, inputs_transform=self.trainval_transform
            )
        else:
            x_test, labels_test = load_processed_crops_and_labels(
                "test", metadata.PROCESSED_DATA_DIR
            )
            y_test = utils.convert_strings_to_labels(
                labels_test,
                self.char_to_idx,
                length=self.output_dims[0],
                with_start_and_end_tokens=True,
            )
            # no augmentation in the test dataset;
            # only embed crops to required dims;
            self.test_dataset = utils.SupervisedDataset(
                x_test, y_test, inputs_transform=self.transform
            )

    @staticmethod
    def add_to_argparse(parser):
        # placeholder for testing purposes;
        parser.add_argument(
            "--augment",
            action="store_true",
            help="Whether to augment train and val datasets.",
        )
        return parser

    def __repr__(self):
        ans = (
            "IAMLines Dataset\n"
            f"Num classes: {len(self.idx_to_char)}\n"
            f"Dims: {self.input_dims}\n"
            f"Output dims: {self.output_dims}\n"
        )
        return ans


def load_processed_crops_and_labels(split: str, dir_path: Path):
    crops = load_processed_line_crops(split, dir_path)
    labels = load_processed_line_labels(split, dir_path)
    assert len(crops) == len(labels)
    return crops, labels


def load_processed_line_labels(split: str, dir_path: Path):
    """Assumes labels stored in dir_path/split/_labels.json"""
    with open(dir_path / split / "_labels.json") as file:
        labels = json.load(file)
    return labels


def load_processed_line_crops(split: str, dir_path: Path):
    """Assumes crops saved as .png in dir_path/split"""
    crop_filenames = sorted(
        (dir_path / split).glob("*.png"),
        key=lambda filename: int(Path(filename).stem),
    )
    crops = [
        utils.read_image_pil(filename, grayscale=True)
        for filename in crop_filenames
    ]
    return crops


def save_images_and_labels(crops, labels, split: str, dir_path: Path):
    split_path = dir_path / split
    split_path.mkdir(parents=True, exist_ok=True)

    with open(split_path / "_labels.json", "w") as file:
        json.dump(labels, file)
    for idx, crop in enumerate(crops):
        crop.save(split_path / f"{idx}.png")


def generate_line_crops_and_labels(iam: IAM, split: str):
    """Get cropped lines and their text labels."""
    crops, labels = [], []
    for iam_id in iam.ids_by_split[split]:
        labels += iam.line_strings_by_id[iam_id]
        image = iam.load_image(iam_id)
        for region in iam.line_regions_by_id[iam_id]:
            crop = image.crop(
                (region["x1"], region["y1"], region["x2"], region["y2"])
            )
            # resize to a height of 28 to save memory;
            h = crop.size[-1]
            crop.resize(
                (int(d / (h / 28)) for d in crop.size), resample=Image.BILINEAR
            )
            crops.append(crop)
    assert len(crops) == len(labels)
    return crops, labels


class IAMLinesTransforms:
    def __init__(
        self,
        augment=False,
        color_jitter_kwargs=None,
        random_affine_kwargs=None,
    ) -> None:
        if color_jitter_kwargs is None:
            color_jitter_kwargs = {"brightness": (0.8, 1.6)}
        if random_affine_kwargs is None:
            # not too crazy of a transform, so it's fine;
            random_affine_kwargs = {
                "degrees": 1,
                "shear": (-30, 20),
                "interpolation": T.InterpolationMode.BILINEAR,
                "fill": 0,
            }
        pil_transforms = [T.Lambda(resize_crop)]
        if augment:
            pil_transforms += [
                T.ColorJitter(**color_jitter_kwargs),
                T.RandomAffine(**random_affine_kwargs),
            ]
        self.pil_transforms = T.Compose(pil_transforms)
        self.pil_to_tensor = T.ToTensor()

    def __call__(self, img):
        img = self.pil_transforms(img)
        return self.pil_to_tensor(img)


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

    iam_lines_dataset, args = mock_lit_dataset(IAMLines)
    iam_lines_dataset.prepare_data()
    iam_lines_dataset.setup(args["stage"])
    print(iam_lines_dataset)
    if args["stage"] == "fit":
        dl = iter(iam_lines_dataset.train_dataloader())
    else:
        dl = iter(iam_lines_dataset.test_dataloader())
    x, y = next(iter(dl))
    print(f"x.shape: {x.shape}, y.shape: {y.shape}")
    print(
        f"x dtype, min, mean, max, std: {(x.dtype, x.min(), x.mean(), x.max(), x.std())}"
    )
    print(f"y dtype, min, max: {(y.dtype, y.min(), y.max())}")
    print(f"batch_size: {len(y)}")
    test_plot(zip(x, y), len(y), min_idx=4)
