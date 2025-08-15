# """Datasets."""
# import os

# import numpy as np

# import torch
# from torch.utils.data import Dataset
# from torchvision import datasets
# from torchvision import transforms


# BASEPATH = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), '..', '..', 'data'))


# class MNIST(datasets.MNIST):
#     """MNIST dataset."""

#     mean_channels = (0.131,)
#     std_channels = (0.308,)

#     transforms = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])

#     def __init__(self, train=True):
#         """MNIST dataset normalized."""
#         super().__init__(
#             BASEPATH, transform=self.transforms, train=train, download=True)

#     def inverse_normalization(self, normalized):
#         """Inverse the normalization applied to the original data.

#         Args:
#             x: Batch of data

#         Returns:
#             Tensor with normalization inversed.

#         """
#         normalized = 0.5 * (normalized + 1)
#         return normalized

import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms

BASEPATH = "/kaggle/input/purification/medmnist/mnist/cw strong"

class MNIST(Dataset):
    """MNIST dataset from pre-saved batch_*.pt files."""

    mean_channels = (0.131,)
    std_channels = (0.308,)

    transforms = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,))
    ])

    def __init__(self, train=True):
        """
        MNIST dataset normalized.
        Loads batch_*.pt files instead of downloading.
        """
        # all batch files are directly inside BASEPATH
        batch_files = sorted(glob.glob(os.path.join(BASEPATH, "batch_*.pt")))
        if not batch_files:
            raise FileNotFoundError(f"No batch_*.pt files found in {BASEPATH}")

        all_imgs, all_labels = [], []
        for f in batch_files:
            batch = torch.load(f)
            if isinstance(batch, dict):
                imgs, labels = batch['images'], batch['labels']
            else:
                imgs, labels = batch  # assume tuple
            all_imgs.append(imgs)
            all_labels.append(labels)

        self.images = torch.cat(all_imgs)
        self.labels = torch.cat(all_labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        if self.transforms:
            img = self.transforms(img)
        return img, label

    def inverse_normalization(self, normalized):
        """Inverse the normalization applied to the original data."""
        return 0.5 * (normalized + 1)
