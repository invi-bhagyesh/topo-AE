"""Datasets."""
import os

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from medmnist import BloodMNIST as _BloodMNIST  

BASEPATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data'))


class BloodMNIST(Dataset):
    """BloodMNIST dataset."""

    mean_channels = (0.131,)
    std_channels = (0.308,)

    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    def __init__(self, train=True):
        """BloodMNIST dataset normalized."""
        self.dataset = _BloodMNIST(
            root=BASEPATH, split='train' if train else 'test', transform=self.transforms, download=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label

    def inverse_normalization(self, normalized):
        """Inverse the normalization applied to the original data.

        Args:
            normalized: Batch of data

        Returns:
            Tensor with normalization inversed.
        """
        return 0.5 * (normalized + 1)
