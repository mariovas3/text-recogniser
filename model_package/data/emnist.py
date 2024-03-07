import json
import random
import shutil
import sys
import zipfile
from pathlib import Path

p = Path(__file__).absolute().parents[1]
if str(p) not in sys.path:
    sys.path.append(str(p))

import h5py
import metadata.emnist as metadata
import numpy as np
import requests
import torchvision.transforms as T
from data.lit_datamodule_abc import BaseDataModule, mock_lit_dataset
from data.utils import SupervisedDataset, split_dataset
from project_utils import change_wd


class EMNIST(BaseDataModule):
    """
    Using the ByClass split of EMNIST.

    EMNIST ByClass:		814,255 characters. 62 unbalanced classes.
    10 classes for digits and 2 * 26 classes for letters.
    """

    def __init__(self, args=None) -> None:
        super().__init__(args)
        self.idx_to_char = metadata.MAPPING
        self.char_to_idx = {c: i for i, c in enumerate(self.idx_to_char)}
        self.transform = T.ToTensor()
        self.input_dims = metadata.DIMS
        self.output_dims = metadata.OUTPUT_DIMS

    def prepare_data(self) -> None:
        if not metadata.PROCESSED_FILE_PATH.exists():
            _download_and_process_emnist(metadata)

    def setup(self, stage):
        if stage == "fit":
            with h5py.File(metadata.PROCESSED_FILE_PATH, "r") as f:
                self.x_train_and_val = f["x_train"][:]
                self.y_train_and_val = f["y_train"][:].squeeze().astype(int)

            data_train_val = SupervisedDataset(
                self.x_train_and_val,
                self.y_train_and_val,
                inputs_transform=self.transform,
            )

            self.train_dataset, self.val_dataset = split_dataset(
                train_frac=metadata.TRAIN_FRAC, dataset=data_train_val, seed=0
            )

        elif stage == "test":
            with h5py.File(metadata.PROCESSED_FILE_PATH, "r") as f:
                self.x_test = f["x_test"][:]
                self.y_test = f["y_test"][:].squeeze().astype(int)
            self.test_dataset = SupervisedDataset(
                self.x_test, self.y_test, inputs_transform=self.transform
            )

    def __repr__(self) -> str:
        return f"EMNIST dataset with {len(self.idx_to_char)} classes and shape is {self.input_dims}.\n"


def _download_and_process_emnist(metadata):
    raw_file_dest = _download_raw_dataset(
        metadata.RAW_URL, metadata.RAW_DATA_DIR, metadata.RAW_FILENAME
    )
    process_byclass_dataset(raw_file_dest, metadata)


def _download_raw_dataset(url, save_destination_dir, save_as_filename):
    # skip mkdir if exists, otherwise also create parents;
    save_destination_dir.mkdir(parents=True, exist_ok=True)
    full_name = save_destination_dir / save_as_filename
    if full_name.exists():
        print(f"\nFILE ALREADY EXISTS!\n{full_name}")
        return full_name
    response = requests.get(url)
    if response.status_code == 200:
        with open(full_name, "wb") as file:
            file.write(response.content)
        print(f"file downloaded successfully to {full_name}")
        return full_name
    print(
        f"Failed download of {full_name}! status code: {response.status_code}"
    )
    return full_name


def _truncate_dataset(x, y, metadata):
    """Limit samples per class to the median count of instances per class."""
    # bincount counts all numbers from 0, ... to max_number;
    # since we offset the labels by NUM_SPECIAL_TOKENS, the first
    # 0, ..., NUM_SPECIAL_TOKENS-1 will have 0 counts and will
    # decrease the median;
    num_special_tokens = metadata.NUM_SPECIAL_TOKENS
    a = y.flatten()
    med_count = int(np.median(np.bincount(a)[num_special_tokens:]))
    print(f"at most {med_count} samples per class;")
    all_sampled_idxs = []
    for label in np.unique(a):
        # idxs of examples of class label;
        idxs = np.where(a == label)[0]
        # shuffle and get first to_get idxs;
        random.shuffle(idxs)
        to_get = min(med_count, len(idxs))
        # append first to_get idxs;
        all_sampled_idxs.append(idxs[:to_get])
    all_sampled_idxs = np.concatenate(all_sampled_idxs)
    return x[all_sampled_idxs], y[all_sampled_idxs]


def process_byclass_dataset(raw_file_dest, metadata):
    dirname = raw_file_dest.parent
    filename = str(raw_file_dest).split("/")[-1]
    PROCESSED_DATA_DIR = metadata.PROCESSED_DATA_DIR
    PROCESSED_DATA_FILE = metadata.PROCESSED_DATA_FILE
    NUM_SPECIAL_TOKENS = metadata.NUM_SPECIAL_TOKENS
    TRUNCATE_DATASET = metadata.TRUNCATE_DATASET

    print("Extracting emnist-byclass.mat")
    with change_wd(dirname):
        # extract from zip file;
        with zipfile.ZipFile(filename) as zf:
            zf.extract("matlab/emnist-byclass.mat")

        # reading matlab data;
        from scipy.io import loadmat

        print("Loading training data from .mat file...")
        data = loadmat("matlab/emnist-byclass.mat")
        x_train = (
            data["dataset"]["train"][0, 0]["images"][0, 0]
            .reshape(-1, 28, 28)
            .swapaxes(1, 2)
        )
        y_train = (
            data["dataset"]["train"][0, 0]["labels"][0, 0] + NUM_SPECIAL_TOKENS
        )
        x_test = (
            data["dataset"]["test"][0, 0]["images"][0, 0]
            .reshape(-1, 28, 28)
            .swapaxes(1, 2)
        )
        y_test = (
            data["dataset"]["test"][0, 0]["labels"][0, 0] + NUM_SPECIAL_TOKENS
        )

        # the dataset is unbalanced, so make each class have
        # at most the median instance count per class;
        if TRUNCATE_DATASET:
            print("Make max count per class the median of counts...")
            x_train, y_train = _truncate_dataset(x_train, y_train, metadata)
            x_test, y_test = _truncate_dataset(x_test, y_test, metadata)

        # save to hdfs format, should be space efficient;
        print("Saving to HDFS in a compressed format...")
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        full_name = PROCESSED_DATA_DIR / PROCESSED_DATA_FILE
        with h5py.File(full_name, "w") as f:
            f.create_dataset(
                "x_train", data=x_train, dtype="u1", compression="lzf"
            )
            f.create_dataset(
                "y_train", data=y_train, dtype="u1", compression="lzf"
            )
            f.create_dataset(
                "x_test", data=x_test, dtype="u1", compression="lzf"
            )
            f.create_dataset(
                "y_test", data=y_test, dtype="u1", compression="lzf"
            )

        # saving json with mapping and input shape;
        print("Saving summary of dataset in json...")
        # the data mapping is in format (idx: ascii_code)
        mapping = {int(k): chr(v) for k, v in data["dataset"]["mapping"][0, 0]}
        characters = metadata.MAPPING
        assert all(v in set(characters) for v in mapping.values())
        essentials = {
            "characters": characters,
            "input_shape": list(x_train.shape[1:]),
        }
        with open(metadata.ESSENTIALS_FILE, "w") as f:
            json.dump(essentials, f)

        # recursively remove the mess from the extraction of the zip;
        print("Cleaning up...")
        shutil.rmtree("matlab")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def test_plot(data_batch, batch_size):
        to_show = min(49, batch_size)
        for i in range(to_show):
            x, y = next(data_batch)
            plt.subplot(7, 7, i + 1)
            plt.imshow(x.squeeze().numpy(), cmap="gray")
            label = metadata.MAPPING[y.item()]
            plt.title(label)
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    emnist_dataset = mock_lit_dataset(EMNIST)
    emnist_dataset.prepare_data()
    temp = emnist_dataset.args
    emnist_dataset.setup(temp["stage"])
    print(emnist_dataset)
    if temp["stage"] == "fit":
        dl = iter(emnist_dataset.train_dataloader())
    else:
        dl = emnist_dataset.test_dataloader()
    x, y = next(iter(dl))
    print(f"x.shape: {x.shape}, y.shape: {y.shape}")
    print(
        f"x dtype, min, mean, max, std: {(x.dtype, x.min(), x.mean(), x.max(), x.std())}"
    )
    print(f"y dtype, min, max: {(y.dtype, y.min(), y.max())}")
    test_plot(zip(x, y), temp["batch_size"])
