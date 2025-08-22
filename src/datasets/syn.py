"""Datasets."""
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Root folder where synthetic dataset is already stored
BASEPATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src','datasets', 'data'))


class SYN(Dataset):
    """Synthetic dataset with images and labels in filenames."""

    mean_channels = (0.5,)
    std_channels = (0.5,)

    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    def __init__(self, train=True):
        """
        Args:
            train (bool): Whether to load the training set or test set.
        """
        # Use subfolders 'train' and 'test' inside BASEPATH
        self.data_dir = os.path.join(BASEPATH, 'out')
        self.image_files = os.listdir(self.data_dir)
        self.transform = self.transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert('RGB')  # convert to 3 channels

        # Extract label from filename: assuming "index_name_label.png"
        label = int(img_name.split('_')[-1].split('.')[0])

        if self.transform:
            image = self.transform(image)

        return image, label

    def inverse_normalization(self, normalized):
        """Inverse the normalization applied to the original data."""
        return 0.5 * (normalized + 1)
