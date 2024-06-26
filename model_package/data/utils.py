import zipfile

import numpy as np
import smart_open
from PIL import Image
from torch import Generator
from torch.utils.data import Dataset, random_split

from model_package.project_utils import change_wd


def hstack_line_crops(line_crops, h_pad=0):
    H, W = 0, 0
    for _ in line_crops:
        w, h = _.size
        H += h + h_pad
        if w > W:
            W = w

    out = Image.new("L", (W, H))
    curr_h = 0
    for crop in line_crops:
        out.paste(crop, box=(0, curr_h))
        curr_h += crop.size[-1] + h_pad
    return out


def resize_crop(crop: Image, new_width, new_height):
    # crop is PIL.image of dtype="L" (so values range from 0 -> 255)
    image = Image.new("L", (new_width, new_height))

    # Resize crop;
    cur_width, cur_height = crop.size
    new_crop_width = round(cur_width * (new_height / cur_height))
    assert new_crop_width <= new_width
    crop = crop.resize((new_crop_width, new_height), resample=Image.BILINEAR)
    image.paste(crop)
    return image


def resize_to_pix_per_line(img, num_lines, pix=28):
    h = img.size[-1]
    scale_factor = h / (pix * num_lines)
    return img.resize(
        (round(d / scale_factor) for d in img.size), resample=Image.BILINEAR
    )


class SupervisedDataset(Dataset):
    def __init__(
        self,
        inputs,
        targets,
        inputs_transform=None,
        targets_transform=None,
    ):
        assert len(inputs) == len(targets)
        self.inputs = inputs
        self.targets = targets
        self.inputs_transform = inputs_transform
        self.targets_transform = targets_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x, y = self.inputs[idx], self.targets[idx]
        if self.inputs_transform is not None:
            x = self.inputs_transform(x)
        if self.targets_transform is not None:
            y = self.targets_transform(y)
        return x, y


def split_dataset(train_frac, dataset, seed):
    """Split dataset in 2 in a reproducible way."""
    size_first = int(train_frac * len(dataset))
    size_second = len(dataset) - size_first
    return random_split(
        dataset,
        (size_first, size_second),
        generator=Generator().manual_seed(seed),
    )


def convert_strings_to_labels(
    strings, char_to_idx, length, with_start_and_end_tokens: bool
):
    """Make each string into array of idxs based on char_to_idx mapping."""
    # init all as pad idxs;
    labels = (
        np.ones((len(strings), length), dtype=np.int64) * char_to_idx["<PAD>"]
    )
    offset = int(with_start_and_end_tokens)
    for i, s in enumerate(strings):
        tokens = list(s)
        # tokens guaranteed to be at most length-2 long;
        # so we can add start and end tokens;
        for ii, token in enumerate(tokens):
            labels[i, ii + offset] = char_to_idx[token]
        if with_start_and_end_tokens:
            labels[i, 0] = char_to_idx["<START>"]
            labels[i, ii + offset + 1] = char_to_idx["<END>"]
    return labels


def resize_image(image: Image.Image, scale_factor):
    """
    Resize image (W, H) -> (W // scale_factor, H // scale_factor).

    Resampling mechanism is PIL.Image.BILINEAR
    """
    if scale_factor == 1:
        return image
    return image.resize(
        (image.width // scale_factor, image.height // scale_factor),
        resample=Image.BILINEAR,
    )


def read_image_pil(image_uri, grayscale=False):
    with smart_open.open(image_uri, "rb") as image_file:
        return read_image_pil_file(image_file, grayscale)


def read_image_pil_file(image_file, grayscale=False):
    with Image.open(image_file) as image:
        if grayscale:
            image = image.convert(mode="L")
        else:
            image = image.convert(image.mode)
        return image


def extract_raw_dataset(filename, dirname):
    print(f"Extracting {filename}...")
    with change_wd(dirname):
        with zipfile.ZipFile(filename, "r") as zf:
            zf.extractall()
