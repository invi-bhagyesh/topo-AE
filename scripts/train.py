#!/usr/bin/env python3
"""Topological Autoencoder inference script (MNIST, CIFAR, FashionMNIST, EMNIST)"""

import torch
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

import cv2
import os
import glob
import numpy as np
from typing import List, Dict, Tuple, Any
import cv2
import os
import glob
import numpy as np
from typing import List, Dict, Tuple, Any

def split_characters(dataloader, padding: int = 10, char_size: int = 64, debug: bool = False, debug_output_dir: str = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Split character images from dataloader into individual characters.
    
    Args:
        dataloader: Iterable that yields (image_array, filename, label) tuples
        padding: Padding to add around each character
        char_size: Uniform size for character images (width and height) 
        debug: Save debug images showing detected character boundaries
        debug_output_dir: Directory to save debug images (required if debug=True)
    
    Returns:
        Tuple of (character_dataloader_list, metadata_dict)
        - character_dataloader_list: List of dicts with keys 'image', 'filename', 'char_label', 'char_index'
        - metadata_dict: Dictionary containing all metadata needed for combining
    """
    
    if debug and debug_output_dir is None:
        raise ValueError("debug_output_dir must be provided when debug=True")
    
    if debug:
        os.makedirs(debug_output_dir, exist_ok=True)
    
    # Convert dataloader to list for two-pass processing
    data_list = list(dataloader)
    
    # First pass: compute global max width and height of all valid contours in dataset
    global_max_w = 0
    global_max_h = 0

    for img_array, filename, label in data_list:
        # Convert numpy array to opencv format if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_color = img_array.astype(np.uint8)
        else:
            img_color = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        thresh = 255 - gray
        _, thresh_bin = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

        valid_contours = []
        for ctr in contours:
            x, y, w, h = cv2.boundingRect(ctr)
            if w > 2 and h > 2:
                valid_contours.append(ctr)

        for i, ctr in enumerate(valid_contours):
            if i >= len(label):
                break
            x, y, w, h = cv2.boundingRect(ctr)
            if w > global_max_w:
                global_max_w = w
            if h > global_max_h:
                global_max_h = h

    print(f"Global max character size determined: width={global_max_w}, height={global_max_h}")

    pad = 5
    character_dataloader = []
    all_files_metadata = {}

    for img_array, filename, label in data_list:
        print(f"Processing {filename}...")

        # Convert numpy array to opencv format if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_color = img_array.astype(np.uint8)
        else:
            img_color = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        thresh = 255 - gray
        _, thresh_bin = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

        valid_contours = []
        for ctr in contours:
            x, y, w, h = cv2.boundingRect(ctr)
            if w > 2 and h > 2:
                valid_contours.append(ctr)

        print(f" -> Found {len(valid_contours)} valid contours for '{label}' (expected {len(label)})")

        metadata = []
        h_orig, w_orig = gray.shape[:2]
        metadata.append(f"ORIGINAL_SIZE {h_orig} {w_orig}")

        for i, ctr in enumerate(valid_contours):
            x, y, w, h = cv2.boundingRect(ctr)
            if i >= len(label):
                break

            char_img = img_color[y:y+h, x:x+w].copy()

            # Create white image of size (global_max_h, global_max_w)
            char_canvas = np.full((global_max_h, global_max_w, 3), 255, dtype=np.uint8)

            # Compute top-left corner to center the char_img in char_canvas
            y_offset = (global_max_h - h) // 2
            x_offset = (global_max_w - w) // 2

            char_canvas[y_offset:y_offset+h, x_offset:x_offset+w] = char_img

            # Add 5 pixel padding border around the image
            char_img_padded = cv2.copyMakeBorder(char_canvas, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[255,255,255])

            char_label = label[i]
            
            # Add to character dataloader
            character_dataloader.append({
                'image': char_img_padded,
                'filename': f"{filename}_{i}_{char_label}",
                'char_label': char_label,
                'char_index': i,
                'original_filename': filename
            })

            # Save global_max_w, global_max_h, and padding in metadata
            metadata.append(f"CHAR {i} {x} {y} {w} {h} {global_max_w} {global_max_h} {pad}")

        if debug:
            debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for i, ctr in enumerate(valid_contours):
                if i < len(label):
                    x, y, w, h = cv2.boundingRect(ctr)
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
                    cv2.putText(debug_img, label[i], (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            cv2.imwrite(os.path.join(debug_output_dir, f"debug_{filename}.png"), debug_img)

        # Store metadata for this file
        base_name = filename.replace('.png', '').replace('.jpg', '')
        all_files_metadata[base_name] = {
            'original_size': (h_orig, w_orig),
            'chars': []
        }
        
        for line in metadata[1:]:  # Skip the ORIGINAL_SIZE line
            parts = line.split()
            if len(parts) == 9 and parts[0] == "CHAR":
                idx, x, y, w, h, max_w, max_h, pad_val = map(int, parts[1:])
                all_files_metadata[base_name]['chars'].append((idx, x, y, w, h, max_w, max_h, pad_val))

    print("✅ Done! Characters processed with uniform max size and padding. Metadata includes max_w, max_h, and padding.")
    final_size_w = global_max_w + pad * 2
    final_size_h = global_max_h + pad * 2
    print(f"🔎 Max character size (before padding): width={global_max_w}, height={global_max_h}")
    print(f"🖼️ Final saved character image size (with padding): width={final_size_w}, height={final_size_h}")

    return character_dataloader, all_files_metadata


def combine_characters(character_dataloader: List[Dict[str, Any]], metadata: Dict[str, Any]) -> List[Tuple[np.ndarray, str]]:
    """
    Combine character images back into original form using metadata.
    
    Args:
        character_dataloader: List of character data from split_characters
        metadata: Metadata dictionary from split_characters
    
    Returns:
        List of (combined_image_array, filename) tuples
    """
    
    print("Processing character combination...")
    
    # Group character data by original filename
    chars_by_file = {}
    for char_data in character_dataloader:
        orig_filename = char_data['original_filename']
        base_name = orig_filename.replace('.png', '').replace('.jpg', '')
        if base_name not in chars_by_file:
            chars_by_file[base_name] = []
        chars_by_file[base_name].append(char_data)
    
    combined_images = []
    
    for base_name, file_metadata in metadata.items():
        print(f"Processing {base_name}...")

        original_size = file_metadata['original_size']
        char_metadata = file_metadata['chars']

        if original_size is None:
            continue

        h_orig, w_orig = original_size
        combined_img = np.full((h_orig, w_orig, 3), 255, dtype=np.uint8)

        # Get character data for this file
        file_chars = chars_by_file.get(base_name, [])
        
        for idx, x, y, w, h, max_w, max_h, pad in char_metadata:
            # Find the corresponding character image
            char_img_padded = None
            for char_data in file_chars:
                if char_data['char_index'] == idx:
                    char_img_padded = char_data['image']
                    break
            
            if char_img_padded is None:
                continue

            # Remove padding from padded char image (now directly the original char size)
            char_img_cropped = char_img_padded[pad:-pad, pad:-pad]

            # Crop center region corresponding to original bounding box (w, h)
            start_y = (char_img_cropped.shape[0] - h) // 2
            start_x = (char_img_cropped.shape[1] - w) // 2
            char_img = char_img_cropped[start_y:start_y+h, start_x:start_x+w]

            combined_img[y:y+h, x:x+w] = char_img

        combined_images.append((combined_img, f"{base_name}.png"))
        print(f"✅ Combined image: {base_name}.png")

    print("✅ Done! All images combined back to original form.")
    return combined_images


# Adapter for FlatImageDataset
class FlatImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Example: dummy label = 0
        label = 0
        return image, label

def create_split_dataloader(dataset):
    """
    Adapter function to convert FlatImageDataset to the format needed by split_characters
    Expects filename format: prefix_label_suffix.jpg where label contains the characters
    """
    for i in range(len(dataset)):
        image, _ = dataset[i]  # Get PIL image
        filename = dataset.image_files[i]
        
        # Convert PIL to numpy array (OpenCV format)
        image_array = np.array(image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Extract label from filename (assuming format: prefix_label_suffix.jpg)
        parts = filename.split("_")
        if len(parts) >= 2:
            label = parts[1]  # This should contain the character sequence
        else:
            # If no underscore format, use filename without extension as label
            label = filename.replace('.jpg', '')
        
        yield (image_array, filename, label)

# PyTorch Dataset for Character Data
class CharacterDataset(Dataset):
    def __init__(self, character_data, transform=None):
        self.character_data = character_data
        self.transform = transform
    
    def __len__(self):
        return len(self.character_data)
    
    def __getitem__(self, idx):
        char_info = self.character_data[idx]
        # Convert OpenCV image (BGR) to PIL (RGB)
        image = cv2.cvtColor(char_info['image'], cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        # Convert character label to numeric (customize as needed)
        char_label = char_info['char_label']
        if char_label.isdigit():
            label = int(char_label)
        elif char_label.isalpha():
            # Convert letters: A=0, B=1, etc. (uppercase)
            label = ord(char_label.upper()) - ord('A')
        else:
            # For special characters, you might want a different mapping
            label = 0  # default
        
        return image, label

# Complete workflow function
def process_characters_through_model(dataset, model, device, output_dir, 
                                   padding=10, char_size=64, batch_size=32, debug=True):
    """
    Complete workflow: split characters -> create dataset -> extract latents
    """
    
    print("Step 1: Splitting images into characters...")
    # Split characters
    character_data, metadata = split_characters(
        create_split_dataloader(dataset), 
        padding=padding, 
        char_size=char_size, 
        debug=debug, 
        debug_output_dir=os.path.join(output_dir, "debug_output") if debug else None
    )
    
    print(f"Found {len(character_data)} individual characters")
    
    print("Step 2: Creating character dataset and dataloader...")
    # Create character dataset
    char_dataset = CharacterDataset(character_data, transform=dataset.transform)
    char_dataloader = DataLoader(
        char_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        drop_last=False,
        num_workers=2 if not os.path.exists('/kaggle') else 0
    )
    
    print("Step 3: Extracting latents from individual characters...")
    # Extract latents from characters
    try:
        char_latents, char_labels = get_space(
            model,
            char_dataloader,
            mode='latent',
            device=device
        )
        print(f"Character latent space shape: {char_latents.shape}")
        print(f"Character labels shape: {char_labels.shape}")
    except Exception as e:
        print(f"Error extracting character latents: {e}")
        return None, None, None, None
    
    print("Step 4: Extracting character reconstructions...")
    # Extract reconstructed character images
    char_original_images, char_reconstructed_images = extract_reconstructions(
        model, char_dataloader, device
    )
    
    print(f"Character original images shape: {char_original_images.shape}")
    print(f"Character reconstructed images shape: {char_reconstructed_images.shape}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Step 5: Saving character results...")
    # Save character data
    char_npz_path = os.path.join(output_dir, "characters_complete.npz")
    np.savez_compressed(
        char_npz_path,
        clean=char_original_images,
        reconstructed=char_reconstructed_images,
        label=char_labels,
        latent=char_latents,
        character_info=[{
            'char_label': char['char_label'],
            'char_index': char['char_index'],
            'original_filename': char['original_filename'],
            'filename': char['filename']
        } for char in character_data]  # Preserve character metadata
    )
    print(f"Saved character data to {char_npz_path}")
    
    # Save character latents CSV
    char_csv_path = os.path.join(output_dir, "characters_latents.csv")
    char_df = pd.DataFrame(char_latents)
    char_df['label'] = char_labels
    char_df['char_label'] = [char['char_label'] for char in character_data]
    char_df['original_filename'] = [char['original_filename'] for char in character_data]
    char_df['char_index'] = [char['char_index'] for char in character_data]
    char_df.to_csv(char_csv_path, index=False)
    print(f"Saved character latents to {char_csv_path}")
    
    # Save character reconstruction statistics
    char_mse_per_sample = np.mean((char_original_images - char_reconstructed_images) ** 2, axis=(1, 2, 3))
    char_stats_path = os.path.join(output_dir, "characters_reconstruction_stats.csv")
    char_stats_df = pd.DataFrame({
        'sample_idx': range(len(char_mse_per_sample)),
        'label': char_labels,
        'char_label': [char['char_label'] for char in character_data],
        'original_filename': [char['original_filename'] for char in character_data],
        'char_index': [char['char_index'] for char in character_data],
        'mse_reconstruction_error': char_mse_per_sample
    })
    char_stats_df.to_csv(char_stats_path, index=False)
    print(f"Saved character reconstruction statistics to {char_stats_path}")
    
    print(f"\nCharacter processing summary:")
    print(f"  - Total characters: {len(char_latents)}")
    print(f"  - Latent dimension: {char_latents.shape[1]}")
    print(f"  - Mean reconstruction MSE: {np.mean(char_mse_per_sample):.6f}")
    print(f"  - Unique character labels: {len(set(char['char_label'] for char in character_data))}")
    
    return character_data, metadata, char_latents, char_labels

# Path setup for different environments
def setup_paths():
    """Setup Python paths for different environments (Kaggle vs local)."""
    if os.path.exists('/kaggle'):
        print("Running in Kaggle environment")
        src_path = Path(__file__).resolve().parent.parent
        if not (src_path / 'src').exists():
            raise ImportError(f"Could not find src directory at {src_path / 'src'}")
        sys.path.insert(0, str(src_path))       
        # In Kaggle, you might need to install the package or copy source files
        # For now, we'll assume the source files are available
    else:
        print("Running in local environment")
        src_path = Path(__file__).resolve().parent.parent
        if not (src_path / 'src').exists():
            raise ImportError(f"Could not find src directory at {src_path / 'src'}")
        sys.path.insert(0, str(src_path))

setup_paths()

try:
    from src.models.approx_based import TopologicallyRegularizedAutoencoder
    from src.models.submodules import DeepAE
    from src.evaluation.utils import get_space
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure the src directory is in your Python path or copy the required modules")
    sys.exit(1)

def download_dataset(name="MNIST", data_dir="./data", train=True, test=True):
    """
    Download dataset using torchvision.
    Args:
        name: Dataset name ("MNIST", "FashionMNIST", "CIFAR", "EMNIST")
        data_dir: Directory to save data
        train, test: Whether to download splits
    Returns:
        dict of datasets {"train": train_set, "test": test_set}
    """
    print(f"Downloading {name} dataset...")

    if name in ["MNIST", "FashionMNIST", "EMNIST"]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif name == "CIFAR":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # RGB
        ])
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    datasets = {}

    if train:
        if name == "MNIST":
            datasets["train"] = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        elif name == "FashionMNIST":
            datasets["train"] = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        elif name == "EMNIST":
            datasets["train"] = torchvision.datasets.EMNIST(data_dir, split="letters", train=True, download=True, transform=transform)
        elif name == "CIFAR":
            datasets["train"] = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)

    if test:
        if name == "MNIST":
            datasets["test"] = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        elif name == "FashionMNIST":
            datasets["test"] = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
        elif name == "EMNIST":
            datasets["test"] = torchvision.datasets.EMNIST(data_dir, split="letters", train=False, download=True, transform=transform)
        elif name == "CIFAR":
            datasets["test"] = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

    return datasets

import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class FlatImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)

        # Load image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Example: dummy label = 0
        label = 0  

        return image, label


def extract_latents_and_reconstructions(
    model_path,
    output_dir,
    data_dir="./data",
    batch_size=126,
    device='cpu',
    dataset_name="MNIST",
    process_train=True,
    process_test=True
):
    """
    Extract latent representations and reconstructed images from dataset.
    
    Args:
        model_path: Path to pre-trained model
        output_dir: Directory to save results
        data_dir: Directory for data
        batch_size: Batch size for processing
        device: Device to run inference on
        process_train: Whether to process training set
        process_test: Whether to process test set
    """
    
    print(f"Loading pre-trained {dataset_name} Topological Autoencoder...")
    
    # 1. Load pre-trained model with exact dataset configuration
    
    if dataset_name == 'MNIST':
        model = TopologicallyRegularizedAutoencoder(
            autoencoder_model='DeepAE',  # Default MNIST model
            lam=0.5002972000959738,     # Default MNIST lambda
            toposig_kwargs={'match_edges': 'symmetric'}  # Default MNIST topology
        )
    elif dataset_name == 'CIFAR' : # Added dataset parser here 
        model = TopologicallyRegularizedAutoencoder(  
                        ae_kwargs ={
                    'input_dims': [
                    3,
                    32,
                    32
                    ]
                },
                    autoencoder_model= "DeepAE",
                    lam= 1.6280214927932581,
                    toposig_kwargs= {
                        "match_edges": "symmetric"
                        }
          )
    elif dataset_name == 'SYN':
        model = TopologicallyRegularizedAutoencoder(  
                        ae_kwargs ={
                    'input_dims': [
                    3,
                    28,
                    44
                    ]
                },
                    autoencoder_model= "DeepAE",
                    lam= 1.6280214927932581,
                    toposig_kwargs= {
                        "match_edges": "symmetric"
                        }
          )

    elif dataset_name == 'FashionMNIST':
        model = TopologicallyRegularizedAutoencoder(
            autoencoder_model='DeepAE',
            lam=1.6280214927932581,
            toposig_kwargs={'match_edges': 'symmetric'}
        )

    elif dataset_name == "EMNIST":
        model = TopologicallyRegularizedAutoencoder(
            autoencoder_model='DeepAE',  # Default MNIST model
            lam=0.5002972000959738,     # Default MNIST lambda
            toposig_kwargs={'match_edges': 'symmetric'}  # Default MNIST topology
        )

    ### This is the dataloading part starts here
    # Load the trained weights
    print(f"Loading model weights from {model_path}")
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to CUDA")
    elif device == 'cuda':
        print("CUDA requested but not available, using CPU")
        device = 'cpu'
    
    # 2. Download dataset
    # datasets = download_dataset(dataset_name, data_dir, train=process_train, test=process_test)

    # transform = transforms.Compose([
    #     transforms.Resize((128, 128)),   # resize to model input size
    #     transforms.ToTensor(),           # convert to tensor
    # ])

    # Your existing dataset
    dataset = FlatImageDataset("/kaggle/input/test-adv-splitted/train_original/train_original")

    print("SPlittinggg !!")
    # Complete workflow: split -> create dataset -> extract latents
    character_data, metadata, char_latents, char_labels = process_characters_through_model(
        dataset=dataset,
        model=model, 
        device=device,
        output_dir="character_results",
        padding=10,
        char_size=64,
        batch_size=32,
        debug=True
    )

    print("Combingin !!")
    # Optional: Reconstruct original images from processed characters
    if character_data and metadata:
        combined_images = combine_characters(character_data, metadata)
        os.makedirs("reconstructed_originals", exist_ok=True)
        for img_array, filename in combined_images:
            cv2.imwrite(f"reconstructed_originals/{filename}", img_array)
    print("DOne")
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # if True:
    #     # Create dataloader
    #     dataloader = DataLoader(
    #         dataset,
    #         batch_size=batch_size,
    #         shuffle=False,
    #         drop_last=False,
    #         num_workers=2 if not os.path.exists('/kaggle') else 0  # No multiprocessing in Kaggle
    #     )
        
    #     # Extract latent representations using existing codebase function
    #     print("Extracting latent representations...")
    #     try:
    #         latents, labels = get_space(
    #             model,
    #             dataloader,
    #             mode='latent',
    #             device=device
    #         )
    #         print(f"Latent space shape: {latents.shape}")
    #         print(f"Labels shape: {labels.shape}")

    #     except Exception as e:
    #         print(f"Error extracting latents with get_space: {e}")
    #         print("Falling back to manual extraction...")
    #         latents, labels = extract_manually(model, dataloader, device)
    #         if dataset_name == "EMNIST":
    #             labels = labels - 1
        
    #     # Extract reconstructed images
    #     print("Extracting reconstructed images...")
    #     original_images, reconstructed_images = extract_reconstructions(
    #         model, dataloader, device
    #     )
        

    #     print(f"Original images shape (rescaled to [0,1]): {original_images.shape}")
    #     print(f"Reconstructed images shape (rescaled to [0,1]): {reconstructed_images.shape}")
        
    #     # Save results in requested format: (clean, reconstructed, label, latent)
    #     print("Saving results...")
        
    #     # Save complete data to NPZ in requested format
    #     npz_path = os.path.join(output_dir, f"{dataset_name.lower()}_{split_name}_complete.npz")
    #     np.savez_compressed(
    #         npz_path,
    #         clean=original_images,           # Original/clean images
    #         reconstructed=reconstructed_images,  # Reconstructed images
    #         label=labels,                    # Labels
    #         latent=latents                   # Latent representations
    #     )
    #     print(f"Saved complete data to {npz_path}")
        
    #     # Also save as separate CSV for latents (for easy analysis)
    #     csv_path = os.path.join(output_dir, f"{dataset_name.lower()}_{split_name}_latents.csv")
    #     df = pd.DataFrame(latents)
    #     df['label'] = labels
    #     df.to_csv(csv_path, index=False)
    #     print(f"Saved latents to {csv_path}")
        
    #     # Save reconstruction error statistics
    #     mse_per_sample = np.mean((original_images - reconstructed_images) ** 2, axis=(1, 2, 3))
    #     stats_path = os.path.join(output_dir, f"{dataset_name.lower()}_{split_name}_reconstruction_stats.csv")
    #     stats_df = pd.DataFrame({
    #         'sample_idx': range(len(mse_per_sample)),
    #         'label': labels,
    #         'mse_reconstruction_error': mse_per_sample
    #     })
    #     stats_df.to_csv(stats_path, index=False)
    #     print(f"Saved reconstruction statistics to {stats_path}")
        
    #     print(f"Successfully processed {split_name} set:")
    #     print(f"  - Samples: {len(latents)}")
    #     print(f"  - Latent dimension: {latents.shape[1]}")
    #     print(f"  - Mean reconstruction MSE: {np.mean(mse_per_sample):.6f}")
    #     print(f"  - Std reconstruction MSE: {np.std(mse_per_sample):.6f}")
        
    #     print(f"\nSaved data format:")
    #     print(f"  - clean: {original_images.shape} (original images)")
    #     print(f"  - reconstructed: {reconstructed_images.shape} (reconstructed images)")
    #     print(f"  - label: {labels.shape} (digit labels 0-9)")
    #     print(f"  - latent: {latents.shape} (latent representations)")
        ### This is the dataloading part
def extract_manually(model, dataloader, device):
    
    """Manually extract latents if get_space function fails."""
    all_latents = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Extracting latents")):
            if device == 'cuda':
                images = images.cuda()
            
            # Extract latent representation
            latents = model.encode(images)
            
            # Convert to numpy
            latents_np = latents.detach().cpu().numpy()
            labels_np = labels.numpy()
            
            all_latents.append(latents_np)
            all_labels.append(labels_np)
    
    # Concatenate all batches
    latents = np.concatenate(all_latents, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    # Shift EMNIST labels from 1–26 to 0–25
    if "EMNIST" in dataloader.dataset.__class__.__name__ or "emnist" in str(dataloader.dataset.__class__).lower():
        labels = labels - 1
    
    return latents, labels

def extract_reconstructions(model, dataloader, device):
    """Extract original and reconstructed images."""
    all_originals = []
    all_reconstructions = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc="Extracting reconstructions")):
            if device == 'cuda':
                images = images.cuda()
            
            # Get reconstruction
            latents = model.encode(images)
            reconstructions = model.decode(latents)
            
            # Convert to numpy
            images_np = images.detach().cpu().numpy()
            reconstructions_np = reconstructions.detach().cpu().numpy()
            
            all_originals.append(images_np)
            all_reconstructions.append(reconstructions_np)
    
    # Concatenate all batches
    original_images = np.concatenate(all_originals, axis=0)
    reconstructed_images = np.concatenate(all_reconstructions, axis=0)
    
    return original_images, reconstructed_images

def main():
    """Main function for running inference."""
    dataset_name = "SYN"
    # Configuration - modify these paths for your setup
    if os.path.exists('/kaggle'):
        # Kaggle environment
        model_path = "/kaggle/input/fawa_topo_ae/pytorch/default/2/Model State.pth"
        output_dir = f"/kaggle/working/{dataset_name.lower()}_inference_output"
        data_dir = f"/kaggle/working/{dataset_name.lower()}_data"
    else:
        # Local environment
        model_path = "/home/aravinthakshan/Projects/mrm/topological-autoencoders/scripts/model_state.pth"
        output_dir = f"/home/aravinthakshan/Projects/mrm/topological-autoencoders/scripts/{dataset_name.lower()}_output"
        data_dir = f"./{dataset_name.lower()}_data"
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Run inference
    extract_latents_and_reconstructions(
        model_path=model_path,
        output_dir=output_dir,
        data_dir=data_dir,
        batch_size=126,
        device=device,
        process_train=True,
        process_test=True,
        dataset_name=dataset_name
    )
    
    print(f"\nInference completed! Results saved to {output_dir}")
    print("\nGenerated files:")
    print(f"- {dataset_name.lower()}_train_complete.npz: Training set with (clean, reconstructed, label, latent)")
    print(f"- {dataset_name.lower()}_test_complete.npz: Test set with (clean, reconstructed, label, latent)")
    print(f"- {dataset_name.lower()}_train_latents.csv: Training set latent representations + labels")
    print(f"- {dataset_name.lower()}_test_latents.csv: Test set latent representations + labels") 
    print(f"- {dataset_name.lower()}_*_reconstruction_stats.csv: Reconstruction error statistics")


if __name__ == "__main__":
    main()