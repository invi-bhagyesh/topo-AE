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
    elif dataset_name == 'CIFAR':
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
    datasets = download_dataset(dataset_name, data_dir, train=process_train, test=process_test)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Process each dataset split
    for split_name, dataset in datasets.items():
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name} {split_name} set ({len(dataset)} samples)")
        print(f"{'='*60}")
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=2 if not os.path.exists('/kaggle') else 0  # No multiprocessing in Kaggle
        )
        
        # Extract latent representations using existing codebase function
        print("Extracting latent representations...")
        try:
            latents, labels = get_space(
                model,
                dataloader,
                mode='latent',
                device=device
            )
            if dataset_name == "EMNIST":
                labels = labels - 1
            print(f"Latent space shape: {latents.shape}")
            print(f"Labels shape: {labels.shape}")
        except Exception as e:
            print(f"Error extracting latents with get_space: {e}")
            print("Falling back to manual extraction...")
            latents, labels = extract_manually(model, dataloader, device)
            if dataset_name == "EMNIST":
                labels = labels - 1
        
        # Extract reconstructed images
        print("Extracting reconstructed images...")
        original_images, reconstructed_images = extract_reconstructions(
            model, dataloader, device
        )
        

        print(f"Original images shape (rescaled to [0,1]): {original_images.shape}")
        print(f"Reconstructed images shape (rescaled to [0,1]): {reconstructed_images.shape}")
        
        # Save results in requested format: (clean, reconstructed, label, latent)
        print("Saving results...")
        
        # Save complete data to NPZ in requested format
        npz_path = os.path.join(output_dir, f"{dataset_name.lower()}_{split_name}_complete.npz")
        np.savez_compressed(
            npz_path,
            clean=original_images,           # Original/clean images
            reconstructed=reconstructed_images,  # Reconstructed images
            label=labels,                    # Labels
            latent=latents                   # Latent representations
        )
        print(f"Saved complete data to {npz_path}")
        
        # Also save as separate CSV for latents (for easy analysis)
        csv_path = os.path.join(output_dir, f"{dataset_name.lower()}_{split_name}_latents.csv")
        df = pd.DataFrame(latents)
        df['label'] = labels
        df.to_csv(csv_path, index=False)
        print(f"Saved latents to {csv_path}")
        
        # Save reconstruction error statistics
        mse_per_sample = np.mean((original_images - reconstructed_images) ** 2, axis=(1, 2, 3))
        stats_path = os.path.join(output_dir, f"{dataset_name.lower()}_{split_name}_reconstruction_stats.csv")
        stats_df = pd.DataFrame({
            'sample_idx': range(len(mse_per_sample)),
            'label': labels,
            'mse_reconstruction_error': mse_per_sample
        })
        stats_df.to_csv(stats_path, index=False)
        print(f"Saved reconstruction statistics to {stats_path}")
        
        print(f"Successfully processed {split_name} set:")
        print(f"  - Samples: {len(latents)}")
        print(f"  - Latent dimension: {latents.shape[1]}")
        print(f"  - Mean reconstruction MSE: {np.mean(mse_per_sample):.6f}")
        print(f"  - Std reconstruction MSE: {np.std(mse_per_sample):.6f}")
        
        print(f"\nSaved data format:")
        print(f"  - clean: {original_images.shape} (original images)")
        print(f"  - reconstructed: {reconstructed_images.shape} (reconstructed images)")
        print(f"  - label: {labels.shape} (digit labels 0-9)")
        print(f"  - latent: {latents.shape} (latent representations)")

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
    dataset_name = "EMNIST"
    # Configuration - modify these paths for your setup
    if os.path.exists('/kaggle'):
        # Kaggle environment
        model_path = "/kaggle/input/invi_emnist/pytorch/default/5/model_state.pth"
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