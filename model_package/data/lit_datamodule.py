"""
Currently, supports distributed data parallel but not multi-node training.
"""


import argparse
import os

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

import model_package.metadata.shared as metadata

BATCH_SIZE = 128
CALLING_THREAD_CPU_COUNT = len(os.sched_getaffinity(0))
GPU_COUNT = torch.cuda.device_count()

# default num workers;
DEFAULT_NUM_WORKERS = CALLING_THREAD_CPU_COUNT
# in ddp, training is on each gpu so div by gpu count
if GPU_COUNT:
    DEFAULT_NUM_WORKERS //= GPU_COUNT


def mock_lit_dataset(lit_data_module) -> None:
    """Load EMNISTLines and print info."""
    parser = argparse.ArgumentParser()
    lit_data_module.add_to_argparse(parser)
    parser.add_argument(
        "--stage",
        type=str,
        default="fit",
        help="One of 'fit' or 'test'. Default is 'fit'.",
    )
    args = parser.parse_args()
    dataset = lit_data_module(args)
    return dataset


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
