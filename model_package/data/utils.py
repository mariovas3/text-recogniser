from torch import Generator
from torch.utils.data import Dataset, random_split


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


def split_dataset(frac, dataset, seed):
    """Split dataset in 2 in a reproducible way."""
    size_first = int(frac * len(dataset))
    size_second = len(dataset) - size_first
    return random_split(
        dataset,
        (size_first, size_second),
        generator=Generator().manual_seed(seed),
    )
