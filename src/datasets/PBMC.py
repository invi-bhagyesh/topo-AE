import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

BASEPATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

class PBMC(Dataset):
    """PBMC Multiome dataset."""

    def __init__(self, train=True):
        """
        Loads RNA + ATAC from HDF5 file.
        Args:
            train: not used here, kept for compatibility with MNIST style
        """
        h5_path = '/kaggle/input/pbmc/pytorch/v1/1/PBMC_Granulocyte_10k_Dataset.h5'
        with h5py.File(h5_path, 'r') as f:
            # Adjust keys depending on your HDF5 structure
            rna = f['rna'][:]
            atac = f['atac'][:]

        # Optionally normalize
        self.rna = torch.tensor(np.log1p(rna), dtype=torch.float32)
        self.atac = torch.tensor(atac / np.max(atac), dtype=torch.float32)

    def __len__(self):
        return self.rna.shape[0]

    def __getitem__(self, idx):
        # Combine RNA + ATAC as input vector
        x = torch.cat([self.rna[idx], self.atac[idx]])
        return x
