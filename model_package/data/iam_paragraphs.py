import json

import torchvision.transforms as T
from PIL import Image

import model_package.data.utils as utils
import model_package.metadata.iam_paragraphs as metadata
from model_package.data.iam import IAM
from model_package.data.lit_datamodule import BaseDataModule


class IAMParagraphs(BaseDataModule):
    def __init__(self, augment=False, **kwargs):
        super().__init__(**kwargs)
        self.augment = augment
        self.idx_to_char = metadata.MAPPING
        self.char_to_idx = {c: i for i, c in enumerate(self.idx_to_char)}
        self.input_dims = metadata.DIMS
        self.output_dims = metadata.OUTPUT_DIMS
        self.transform = ParagraphTransform()
        self.trainval_transform = ParagraphTransform(augment)

    @staticmethod
    def add_to_argparse(parser):
        # placeholder for testing purposes;
        parser.add_argument(
            "--augment",
            action="store_true",
            help="Whether to augment train and val images.",
        )
        return parser

    def prepare_data(self) -> None:
        if metadata.PROCESSED_DATA_DIR.exists():
            print(
                f"PROCESSED DIR already exists: {metadata.PROCESSED_DATA_DIR}"
            )
            return

        print("Preparing IAMParagraphs...")
        metadata.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        iam = IAM()
        iam.prepare_data()

        properties = {}
        for split in ("train", "val", "test"):
            print(f"Preparing {split} paragraphs...")
            crops, labels = get_paragraph_crops_and_labels(iam, split)
            _save_crops_and_labels(crops, labels, split)

            properties.update(
                {
                    iam_id: {
                        # save as (height, width)
                        "crop_shape": crops[iam_id].size[::-1],
                        "par_len": len(label),
                        "num_lines": label.count("\n") + 1,
                    }
                    for iam_id, label in labels.items()
                }
            )

        with open(
            metadata.PROCESSED_DATA_DIR / "_properties.json", "w"
        ) as file:
            # make json pretty by indenting with 4 spaces;
            json.dump(properties, file, indent=4)

    def setup(self, stage) -> None:
        assert stage in ("fit", "test")
        validate_dims()
        if stage == "fit":
            self.train_dataset = self._load_dataset(
                "train", transform=self.trainval_transform
            )
            self.val_dataset = self._load_dataset(
                "val", transform=self.trainval_transform
            )
        else:
            self.test_dataset = self._load_dataset(
                "test", transform=self.transform
            )

    def _load_dataset(self, split: str, transform):
        crops, labels = load_processed_crops_and_labels(split)
        targets = utils.convert_strings_to_labels(
            strings=labels,
            char_to_idx=self.char_to_idx,
            length=self.output_dims[0],
            with_start_and_end_tokens=True,
        )
        return utils.SupervisedDataset(
            crops, targets, inputs_transform=transform
        )

    def __repr__(self):
        ans = (
            "IAMParagraphs Dataset\n"
            f"Num classes: {len(self.idx_to_char)}\n"
            f"Input dims: {self.input_dims}\n"
            f"Output dims: {self.output_dims}\n"
        )
        return ans


class ParagraphTransform:
    def __init__(
        self,
        augment=False,
        color_jitter_kwargs=None,
        random_affine_kwargs=None,
        random_perspective_kwargs=None,
        gaussian_blur_kwargs=None,
        sharpness_kwargs=None,
    ):
        self.pil_to_tensor = T.ToTensor()
        if not augment:
            self.pil_transforms = T.Compose(
                [T.CenterCrop(metadata.IMAGE_SHAPE)]
            )
        else:
            if color_jitter_kwargs is None:
                color_jitter_kwargs = {"brightness": 0.4, "contrast": 0.4}
            if random_affine_kwargs is None:
                random_affine_kwargs = {
                    "degrees": 3,
                    "shear": 6,
                    "scale": (0.95, 1),
                    "interpolation": T.InterpolationMode.BILINEAR,
                }
            if random_perspective_kwargs is None:
                random_perspective_kwargs = {
                    "distortion_scale": 0.2,
                    "p": 0.5,
                    "interpolation": T.InterpolationMode.BILINEAR,
                }
            if gaussian_blur_kwargs is None:
                gaussian_blur_kwargs = {
                    "kernel_size": (3, 3),
                    "sigma": (0.1, 1.0),
                }
            if sharpness_kwargs is None:
                sharpness_kwargs = {"sharpness_factor": 2, "p": 0.5}

            # these are all very mild transforms;
            self.pil_transforms = T.Compose(
                [
                    T.ColorJitter(**color_jitter_kwargs),
                    # this one is tricky because effectively
                    # reduces size of text part and pads a lot;
                    T.RandomCrop(
                        # image_shape is (height, width);
                        size=metadata.IMAGE_SHAPE,
                        padding=None,
                        pad_if_needed=True,
                        fill=0,
                        padding_mode="constant",
                    ),
                    T.RandomAffine(**random_affine_kwargs),
                    T.RandomPerspective(**random_perspective_kwargs),
                    T.GaussianBlur(**gaussian_blur_kwargs),
                    T.RandomAdjustSharpness(**sharpness_kwargs),
                ]
            )

    def __call__(self, img):
        img = self.pil_transforms(img)
        return self.pil_to_tensor(img)


def validate_dims():
    summary = _get_properties()
    assert all(i >= j for i, j in zip(metadata.DIMS[1:], summary["max_crop"]))
    # account for start and end tokens;
    assert metadata.OUTPUT_DIMS[0] >= summary["max_par_len"] + 2


def load_processed_crops_and_labels(split: str):
    split_path = metadata.PROCESSED_DATA_DIR / split
    with open(split_path / "_labels.json", "r") as file:
        labels = json.load(file)

    sorted_ids = sorted(labels.keys())
    ordered_crops = [
        Image.open(split_path / f"{iam_id}.png").convert("L")
        for iam_id in sorted_ids
    ]
    ordered_labels = [labels[iam_id] for iam_id in sorted_ids]
    assert len(ordered_crops) == len(ordered_labels)
    return ordered_crops, ordered_labels


def get_paragraph_crops_and_labels(iam: IAM, split: str, pix=28):
    """
    Get image crops and string labels dictionaries.

    pix (int): resizes paragraphs to have 28 pixel-height lines,
        also adjusts width accordingly.
    """
    crops = {}
    labels = {}
    for iam_id in iam.ids_by_split[split]:
        if iam_id in iam.ignore_paragraph_ids:
            continue
        image = iam.load_image(iam_id)
        region = iam.paragraph_regions_by_id[iam_id]
        crops[iam_id] = image.crop(
            (region["x1"], region["y1"], region["x2"], region["y2"])
        )
        # make crops with height of 28 pixels per line;
        num_lines = len(iam.line_strings_by_id[iam_id])
        crops[iam_id] = utils.resize_to_pix_per_line(
            crops[iam_id], num_lines, pix=pix
        )
        labels[iam_id] = iam.paragraph_string_by_id[iam_id]
    assert len(crops) == len(labels)
    return crops, labels


def _save_crops_and_labels(crops, labels, split) -> None:
    split_dir = metadata.PROCESSED_DATA_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)

    with open(split_dir / "_labels.json", "w") as file:
        # indent for pretty print of json;
        json.dump(labels, file, indent=4)

    for iam_id, crop in crops.items():
        crop.save(split_dir / f"{iam_id}.png")


def _get_properties():
    with open(metadata.PROCESSED_DATA_DIR / "_properties.json", "r") as file:
        properties = json.load(file)

    ans = {
        "max_crop": None,
        "min_crop": None,
        "max_par_len": None,
        "min_par_len": None,
        "max_num_lines": None,
        "min_num_lines": None,
    }
    for i, (iam_id, metrics) in enumerate(properties.items()):
        a_ratio = metrics["crop_shape"][-1] / metrics["crop_shape"][0]
        if i == 0:
            ans["max_crop"] = metrics["crop_shape"]
            ans["min_crop"] = metrics["crop_shape"]
            ans["max_par_len"] = metrics["par_len"]
            ans["min_par_len"] = metrics["par_len"]
            ans["max_num_lines"] = metrics["num_lines"]
            ans["min_num_lines"] = metrics["num_lines"]
            ans["max_aspect_ratio"] = a_ratio
            ans["min_aspect_ratio"] = a_ratio
            ans["max_aspect_ratio_id"] = iam_id
            ans["min_aspect_ratio_id"] = iam_id
        else:
            ans["max_crop"] = [
                max(first, second)
                for first, second in zip(
                    ans["max_crop"], metrics["crop_shape"]
                )
            ]
            ans["min_crop"] = [
                min(first, second)
                for first, second in zip(
                    ans["max_crop"], metrics["crop_shape"]
                )
            ]
            ans["max_par_len"] = max(ans["max_par_len"], metrics["par_len"])
            ans["min_par_len"] = min(ans["min_par_len"], metrics["par_len"])
            ans["max_num_lines"] = max(
                ans["max_num_lines"], metrics["num_lines"]
            )
            ans["min_num_lines"] = min(
                ans["min_num_lines"], metrics["num_lines"]
            )
            if ans["max_aspect_ratio"] < a_ratio:
                ans["max_aspect_ratio"] = a_ratio
                ans["max_aspect_ratio_id"] = iam_id
            if ans["min_aspect_ratio"] > a_ratio:
                ans["min_aspect_ratio"] = a_ratio
                ans["min_aspect_ratio_id"] = iam_id
    return ans


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

    iam_par_dataset, args = mock_lit_dataset(IAMParagraphs)
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
