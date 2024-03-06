"""
Currently, supports distributed data parallel but not multi-node training.
"""


import argparse
import os
import sys
from pathlib import Path

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

p = Path(__file__).absolute().parents[1]
if str(p) not in sys.path:
    sys.path.append(str(p))

import metadata.shared as metadata

BATCH_SIZE = 128
CALLING_THREAD_CPU_COUNT = len(os.sched_getaffinity(0))
GPU_COUNT = torch.cuda.device_count()

# default num workers;
DEFAULT_NUM_WORKERS = CALLING_THREAD_CPU_COUNT
# in ddp, training is on each gpu so div by gpu count
if GPU_COUNT:
    DEFAULT_NUM_WORKERS //= GPU_COUNT


def mock_dataset_creation(data_module_class) -> None:
    """Load EMNISTLines and print info."""
    parser = argparse.ArgumentParser()
    data_module_class.add_to_argparse(parser)
    parser.add_argument(
        "--stage",
        type=str,
        default="fit",
        help="One of 'fit' or 'test'. Default is 'fit'.",
    )
    args = parser.parse_args()
    dataset = data_module_class(args)
    dataset.prepare_data()
    temp = vars(args)
    dataset.setup(temp["stage"])
    print(dataset)
    if temp["stage"] == "fit":
        dl = dataset.train_dataloader()
    else:
        dl = dataset.test_dataloader()
    x, y = next(iter(dl))
    print(f"x.shape: {x.shape}, y.shape: {y.shape}")
    print(
        f"x dtype, min, mean, max, std: {(x.dtype, x.min(), x.mean(), x.max(), x.std())}"
    )
    print(f"y dtype, min, max: {(y.dtype, y.min(), y.max())}")


class BaseDataModule(LightningDataModule):
    """
    Basic usage:

    prepare_data(self) - how to download, tokenize, save, etc.
        Runs once unless prepare_data_per_node is specified.
    setup(self, stage) - how to split, define dataset, etc.
        setup is called from every process across all the nodes.
        Setting state here is recommended.
        There are also data operations you might want to perform on every GPU. Use setup() to do things like:
            * count num classes;
            * build vocab;
            * perform train/val/test splits;
            * create datasets;
            * apply transforms (defined explicitly in datamodule);
        The stage arg separates logic for the lightning
        Trainer's fit, validate, test and predict methods;
    teardown(self) - used to clean up the state;
        Called from every process across all nodes;
    train_dataloader - return train loader;
        Usually just wrap the dataset from the setup stage; it's
        what the Trainer's fit method is gonna use.
    val_dataloader - used by Trainer's fit() and validate() methods.
    test_dataloader - used by Trainer's test() method.
    predict_dataloader - used by Trainer's predict() method.
    """

    def __init__(self, args=None) -> None:
        super().__init__()
        self.args = {} if args is None else vars(args)
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", DEFAULT_NUM_WORKERS)

        self.on_gpu = bool(self.args.get("gpus", 0))

    @classmethod
    def data_dirname(cls):
        return metadata.DATA_DIR

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size",
            type=int,
            default=BATCH_SIZE,
            help=f"Set batch size, default is {BATCH_SIZE}.",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=DEFAULT_NUM_WORKERS,
            help=f"Num processes to load data. Default is {DEFAULT_NUM_WORKERS}.",
        )
        return parser

    def prepare_data(self) -> None:
        """
        Do one-off download, transform and persist to disk.

        Don't modify state (e.g., self.x = y), since will not be visible from other
        processes if run on multiple gpus.
        """

    def setup(self, stage) -> None:
        """
        Get train, val, test, predict datasets depending on stage.

        stage in ('fit', 'test', 'predict').

        This is the place to modify the state (e.g., self.training_dataset = something)
        since is executed for each process for each node.
        """

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )
