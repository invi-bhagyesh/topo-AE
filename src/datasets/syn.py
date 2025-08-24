"""Datasets."""
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

# Root folder where synthetic dataset is already stored
# BASEPATH = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), '..', '..', 'src','datasets', 'data', 'characters'))

#change for differnt dataset
BASEPATH = "/kaggle/input/invi-protego/train_split_generated/train_split_generated"

class SYN(Dataset):
    """Synthetic dataset with images and labels in filenames."""

    def __init__(self, train=True):
        """
        Args:
            train (bool): Whether to load the training set or test set.
        """
        # Use subfolders 'train' and 'test' inside BASEPATH
        self.data_dir = os.path.join(BASEPATH, 'out')
        all_files = os.listdir(self.data_dir)

        # Keep only files with index <= 30000 (index is 3rd part in filename)
        self.image_files = [
            f for f in all_files 
            if int(f.split('_')[2]) <= 30000
        ]

        sample_img = Image.open(os.path.join(self.data_dir, self.image_files[0]))
        channels = 3 if sample_img.mode == "RGB" else 1

        all_pixels = []
        for img_name in self.image_files:
            img = Image.open(os.path.join(self.data_dir, img_name)).convert("RGB" if channels == 3 else "L")
            arr = np.array(img, dtype=np.float32) / 255.0
            if channels == 1:
                arr = arr[..., None]
            all_pixels.append(arr.reshape(-1, channels))

        all_pixels = np.concatenate(all_pixels, axis=0)
        mean = all_pixels.mean(axis=0).tolist()
        std = all_pixels.std(axis=0).tolist()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        
        image = Image.open(img_path).convert("RGB" if len(self.transform.transforms[1].mean) == 3 else "L")

        # Extract label from split filename: "originalWord_index_char.png"
        # 0_pTLTkrRoKu_0_0_p.png
        # 10000_HxhWHgeFom_10000_0_H.png
        ch = img_name.split('_')[-1].split('.')[0]  # get last character

        if ch.islower():
            label = ord(ch) - ord('a')           # 0-25
        elif ch.isupper():
            label = 26 + (ord(ch) - ord('A'))    # 26-51
        else:
            label = -1  # for non-alphabetic characters

        if self.transform:
            image = self.transform(image)

        return image, label
    
   

    def inverse_normalization(self, normalized):
        """Inverse the normalization applied to the original data."""
        return 0.5 * (normalized + 1)
