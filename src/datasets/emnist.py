"""Datasets."""
import os

import torch
from torchvision import datasets
from torchvision import transforms


BASEPATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data'))


class EMNIST(datasets.EMNIST):
    """EMNIST Letters dataset (labels shifted 0–25)."""

    mean_channels = (0.131,)
    std_channels = (0.308,)

    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    def __init__(self, train=True):
        """EMNIST Letters dataset normalized with labels shifted to 0–25."""
        super().__init__(
            BASEPATH, split="letters", transform=self.transforms, train=train, download=True)

    def __getitem__(self, index):
        """Return image and shifted label (0–25 instead of 1–26)."""
        img, target = super().__getitem__(index)
        target = target - 1  # shift labels
        return img, target

    def inverse_normalization(self, normalized):
        """Inverse the normalization applied to the original data."""
        normalized = 0.5 * (normalized + 1)
        return normalized
