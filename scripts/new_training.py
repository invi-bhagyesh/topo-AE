#!/usr/bin/env python3
"""Kaggle-compatible inference script for pre-trained MNIST Topological Autoencoder on adversarial MNIST datasets"""

import torch
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import glob

import sys
from pathlib import Path

# Resolve the src directory two levels up from this script
# src_path = Path(__file__).resolve().parent.parent / 'src'

# if not src_path.exists():
#     raise ImportError(f"Could not find src directory at {src_path}")

# # Add src to Python path
# sys.path.insert(0, str(src_path))


# sys.path.append(str(src_path))
# try:
#     from models.approx_based import TopologicallyRegularizedAutoencoder
#     from models.submodules import DeepAE
#     from evaluation.utils import get_space
# except ImportError as e:
#     print(f"Import error: {e}")
#     print("Please ensure the src directory is in your Python path")
#     sys.exit(1)

# Instead of pointing sys.path to src/, point it to src’s parent
src_path = Path(__file__).resolve().parent.parent  # project root
if not (src_path / 'src').exists():
    raise ImportError(f"Could not find src directory at {src_path / 'src'}")

# Add project root to Python path so "src" is a package
sys.path.insert(0, str(src_path))

try:
    from src.models.approx_based import TopologicallyRegularizedAutoencoder
    from src.models.submodules import DeepAE
    from src.evaluation.utils import get_space
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure the src directory is in your Python path")
    sys.exit(1)


from torch.utils.data import DataLoader

class AdversarialMNISTDataset(torch.utils.data.Dataset):
    """Dataset class for adversarial MNIST data stored as .pt files."""
    
    def __init__(self, data_dir, attack_type=None):
        """
        Args:
            data_dir: Directory containing .pt files (e.g., "cw strong", "fgsm strong")
            attack_type: Specific attack type to load (optional, if None loads all)
        """
        self.data_dir = Path(data_dir)
        print("testing")
        print(self.data_dir, type(data_dir))
        # Find all .pt files in the directory
        if attack_type:
            # Load specific attack type
            pt_files = list(self.data_dir.glob("*.pt"))

        else:
            # Load all .pt files
            pt_files = list(self.data_dir.glob("*.pt"))
        
        if not pt_files:
            raise ValueError(f"No .pt files found in {data_dir}")
        
        # Sort files by batch number for consistent ordering
        pt_files.sort(key=lambda x: int(x.stem.split('_')[1]))
        
        print(f"Found {len(pt_files)} .pt files")
        
        # Load and concatenate all batches
        all_data = []
        all_labels = []
        
        for pt_file in pt_files:
            print(f"Loading {pt_file.name}...")
            batch_data = torch.load(pt_file, map_location='cpu')
            
            # Handle different possible data formats
            if isinstance(batch_data, dict):
                # If it's a dictionary with 'data' and 'labels' keys
                if 'data' in batch_data and 'labels' in batch_data:
                    data = batch_data['data']
                    labels = batch_data['labels']
                elif 'images' in batch_data and 'labels' in batch_data:
                    data = batch_data['images']
                    labels = batch_data['labels']
                else:
                    # Try to infer the structure
                    keys = list(batch_data.keys())
                    print(f"Available keys in {pt_file.name}: {keys}")
                    # Assume first key is data, second is labels
                    data = batch_data[keys[0]]
                    labels = batch_data[keys[1]]
            elif isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                # If it's a tuple/list of (data, labels)
                data, labels = batch_data
            else:
                # Assume it's just the data tensor
                data = batch_data
                # Create dummy labels (you might want to modify this)
                labels = torch.zeros(data.shape[0])
            
            # Convert to numpy if needed
            if torch.is_tensor(data):
                data = data.detach().cpu().numpy()
            if torch.is_tensor(labels):
                labels = labels.detach().cpu().numpy()
            
            all_data.append(data)
            all_labels.append(labels)
        
        # Concatenate all batches
        self.data = np.concatenate(all_data, axis=0)
        self.labels = np.concatenate(all_labels, axis=0)
        
        # Ensure data is in the right format for MNIST model
        if len(self.data.shape) == 3:
            # If (n_samples, height, width), add channel dimension
            self.data = self.data[:, None, :, :]  # (n_samples, 1, height, width)
        elif len(self.data.shape) == 2:
            # If (n_samples, height*width), reshape to (n_samples, 1, height, height)
            height = int(np.sqrt(self.data.shape[1]))
            self.data = self.data.reshape(-1, 1, height, height)
        
        # Normalize to [-1, 1] range (MNIST normalization)
        if self.data.max() > 1.0:
            self.data = self.data / 255.0
        self.data = 2 * self.data - 1  # Scale to [-1, 1]
        
        print(f"Final data shape: {self.data.shape}")
        print(f"Final labels shape: {self.labels.shape}")
        print(f"Data range: [{self.data.min():.3f}, {self.data.max():.3f}]")
        print(f"Unique labels: {np.unique(self.labels)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), self.labels[idx]

def extract_latents_and_reconstructions(
    model_path,
    data_dir,
    output_dir,
    attack_type=None,
    batch_size=126,  # Same as MNIST config
    device='cpu'
):
    """Extract both latent representations AND reconstructed images from pre-trained MNIST model on adversarial dataset."""
    
    print(f"Loading pre-trained MNIST Topological Autoencoder...")
    
    # 1. Load pre-trained model with exact MNIST configuration
    model = TopologicallyRegularizedAutoencoder(
        autoencoder_model='DeepAE',  # Default MNIST model
        lam=0.5002972000959738,     # Default MNIST lambda
        toposig_kwargs={'match_edges': 'symmetric'}  # Default MNIST topology
    )
    
    # Load the trained weights
    print(f"Loading model weights from {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    if device == 'cuda':
        model = model.cuda()
    
    # 2. Create dataset and dataloader
    print(f"Creating dataset from {data_dir}...")
    dataset = AdversarialMNISTDataset(data_dir, attack_type)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    # 3. Extract latent representations using existing codebase function
    print("Extracting latent representations...")
    latent, labels = get_space(
        model,
        dataloader,
        mode='latent',
        device=device
    )
    
    print(f"Latent space shape: {latent.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # 4. Extract reconstructed images
    print("Extracting reconstructed images...")
    all_reconstructions = []
    all_original = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, batch_labels) in enumerate(dataloader):
            if device == 'cuda':
                images = images.cuda()
            
            # Get latent and reconstruction
            latent_batch = model.encode(images)
            reconst_batch = model.decode(latent_batch)
            
            # Convert to numpy
            images_np = images.detach().cpu().numpy()
            reconst_np = reconst_batch.detach().cpu().numpy()
            
            all_original.append(images_np)
            all_reconstructions.append(reconst_np)
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}")
    
    # Concatenate all batches
    original_images = np.concatenate(all_original, axis=0)
    reconstructed_images = np.concatenate(all_reconstructions, axis=0)
    
    print(f"Original images shape: {original_images.shape}")
    print(f"Reconstructed images shape: {reconstructed_images.shape}")
    
    # 5. Save everything
    print("Saving results...")
    
    # Create output directory if it doesn't exist (Kaggle compatible)
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine output filename based on attack type
    if attack_type:
        base_name = f"adversarial_mnist_{attack_type.replace(' ', '_')}"
    else:
        base_name = "adversarial_mnist_all"
    
    # Save to CSV (latents only)
    csv_path = os.path.join(output_dir, f"{base_name}_latents.csv")
    df = pd.DataFrame(latent)
    df['labels'] = labels
    df.to_csv(csv_path, index=False)
    
    # Save to NPZ (everything: latents, labels, original, reconstructed)
    npz_path = os.path.join(output_dir, f"{base_name}_complete.npz")
    np.savez(
        npz_path,
        latents=latent,
        labels=labels,
        original_images=original_images,
        reconstructed_images=reconstructed_images
    )
    
    print(f"Saved latents to {csv_path}")
    print(f"Saved complete data to {npz_path}")
    return latent, labels, original_images, reconstructed_images

def process_all_attacks(
    model_path,
    base_data_dir,
    output_dir="adversarial_mnist_results",
    device='cpu'
):
    """Process all attack types in the base directory."""
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    # Check if data directory exists
    if not os.path.exists(base_data_dir):
        print(f"Error: Data directory not found: {base_data_dir}")
        return
    
    # List available attack types
    attack_dirs = [d for d in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, d))]
    print(f"Available attack types: {attack_dirs}")
    
    # Process each attack type
    for attack_type in attack_dirs:
        attack_data_dir = os.path.join(base_data_dir, attack_type)
        print(f"\n{'='*50}")
        print(f"Processing {attack_type}...")
        print(f"{'='*50}")
        
        try:
            latent, labels, original_images, reconstructed_images = extract_latents_and_reconstructions(
                model_path=model_path,
                data_dir=attack_data_dir,
                output_dir=output_dir,
                attack_type=attack_type,
                device=device
            )
            
            print(f"Successfully processed {attack_type}: {len(latent)} samples")
            print(f"  - Original images: {original_images.shape}")
            print(f"  - Reconstructed images: {reconstructed_images.shape}")
            
        except Exception as e:
            print(f"Error processing {attack_type}: {e}")
            continue
    
    print(f"\nAll processing completed! Results saved to {output_dir}/")

# Example usage functions for Kaggle notebooks
def quick_inference_example():
    """Quick example of how to use the inference script in Kaggle."""
    
    # Example paths - modify these for your Kaggle setup
    model_path = "/home/aravinthakshan/Projects/mrm/topological-autoencoders/scripts/model_state.pth"
    base_data_dir = "mnist"
    output_dir = "/home/aravinthakshan/Projects/mrm/topological-autoencoders/scripts/output"
    
    print("Example usage:")
    print(f"model_path = '{model_path}'")
    print(f"base_data_dir = '{base_data_dir}'")
    print(f"output_dir = '{output_dir}'")
    
    print("\nTo run inference:")
    print("process_all_attacks(model_path, base_data_dir, output_dir)")
    
    print("\nOr for a single attack type:")
    print("extract_latents_and_reconstructions(model_path, base_data_dir, output_dir, 'cw strong')")
    
    print("\nThis will save:")
    print("- CSV file with latent representations (2D)")
    print("- NPZ file with latents, labels, original images, and reconstructed images")

# if __name__ == "__main__":
#     quick_inference_example()

if __name__ == "__main__":

    model_path = "/kaggle/input/mnist_load/pytorch/mnist/2/model_64_mnist.pth"
    base_data_dir = "/kaggle/input/purification/medmnist/mnist"  # or your actual data directory path
    output_dir = "/kaggle/working/output"
    
    process_all_attacks(model_path, base_data_dir, output_dir)