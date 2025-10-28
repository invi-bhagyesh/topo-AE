#!/usr/bin/env python3
"""Inference and visualization script for a standard Autoencoder on clean and adversarial MNIST data."""
import torch
import numpy as np
import os
import sys
from pathlib import Path
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
"""Base class for autoencoder models."""
import abc
from typing import Dict, Tuple

import torch.nn as nn


# class AutoencoderModel(nn.Module, metaclass=abc.ABCMeta):
#     """Abstract base class for autoencoders."""

#     # pylint: disable=W0221
#     @abc.abstractmethod
#     def forward(self, x) -> Tuple[float, Dict[str, float]]:
#         """Compute loss for model.

#         Args:
#             x: Tensor with data

#         Returns:
#             Tuple[loss, dict(loss_component_name -> loss_component)]

#         """

#     @abc.abstractmethod
#     def encode(self, x):
#         """Compute latent representation."""

#     @abc.abstractmethod
#     def decode(self, z):
#         """Compute reconstruction."""
# class View(nn.Module):
#     def __init__(self, shape):
#         super().__init__()
#         self.shape = shape

#     def forward(self, x):
#         return x.view(*self.shape)

# class DeepAE(AutoencoderModel):
#     """1000-500-250-2-250-500-1000."""
#     def __init__(self, input_dims=(1, 28, 28)):
#         super().__init__()
#         self.input_dims = input_dims
#         n_input_dims = np.prod(input_dims)
#         self.encoder = nn.Sequential(
#             View((-1, n_input_dims)),
#             nn.Linear(n_input_dims, 1000),
#             nn.ReLU(True),
#             nn.BatchNorm1d(1000),
#             nn.Linear(1000, 500),
#             nn.ReLU(True),
#             nn.BatchNorm1d(500),
#             nn.Linear(500, 250),
#             nn.ReLU(True),
#             nn.BatchNorm1d(250),
#             nn.Linear(250, 2) # latent dim
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(2, 250), # latent dim
#             nn.ReLU(True),
#             nn.BatchNorm1d(250),
#             nn.Linear(250, 500),
#             nn.ReLU(True),
#             nn.BatchNorm1d(500),
#             nn.Linear(500, 1000),
#             nn.ReLU(True),
#             nn.BatchNorm1d(1000),
#             nn.Linear(1000, n_input_dims),
#             View((-1,) + tuple(input_dims)),
#             nn.Tanh()
#         )
#         self.reconst_error = nn.MSELoss()

#     def encode(self, x):
#         """Compute latent representation using convolutional autoencoder."""
#         return self.encoder(x)

#     def decode(self, z):
#         """Compute reconstruction using convolutional autoencoder."""
#         return self.decoder(z)

#     def forward(self, x):
#         """Apply autoencoder to batch of input images.

#         Args:
#             x: Batch of images with shape [bs x channels x n_row x n_col]

#         Returns:
#             tuple(reconstruction_error, dict(other errors))

#         """
#         latent = self.encode(x)
#         x_reconst = self.decode(latent)
#         reconst_error = self.reconst_error(x, x_reconst)
#         return reconst_error, {'reconstruction_error': reconst_error}

# Autoencoder definition (same as training)
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z).view(-1, 1, 28, 28)
        return x_recon
    def encode(self, x):
        return self.encoder(x)
    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)

def visualize_latents(latents, labels, save_file=None):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    if latents.shape[1] > 2:
        latents = PCA(n_components=2).fit_transform(latents)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='tab10', s=5)
    plt.colorbar(scatter)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("PCA Latent Space Visualization")
    if save_file:
        plt.savefig(save_file, dpi=300)
        plt.close()
    else:
        plt.show()


class AdversarialMNISTDataset(torch.utils.data.Dataset):
    """Dataset class for adversarial MNIST data stored as .pt files."""
    def __init__(self, data_dir, attack_type=None):
        self.data_dir = Path(data_dir)
        if attack_type:
            pt_files = list(self.data_dir.glob("*.pt"))
        else:
            pt_files = list(self.data_dir.glob("*.pt"))
        if not pt_files:
            raise ValueError(f"No .pt files found in {data_dir}")
        pt_files.sort(key=lambda x: int(x.stem.split('_')[1]))
        all_data = []
        all_labels = []
        for pt_file in pt_files:
            batch_data = torch.load(pt_file, map_location='cpu')
            if isinstance(batch_data, dict):
                if 'data' in batch_data and 'labels' in batch_data:
                    data = batch_data['data']
                    labels = batch_data['labels']
                elif 'images' in batch_data and 'labels' in batch_data:
                    data = batch_data['images']
                    labels = batch_data['labels']
                else:
                    keys = list(batch_data.keys())
                    data = batch_data[keys[0]]
                    labels = batch_data[keys[1]]
            elif isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                data, labels = batch_data
            else:
                data = batch_data
                labels = torch.zeros(data.shape[0])
            if torch.is_tensor(data):
                data = data.detach().cpu().numpy()
            if torch.is_tensor(labels):
                labels = labels.detach().cpu().numpy()
            all_data.append(data)
            all_labels.append(labels)
        self.data = np.concatenate(all_data, axis=0)
        self.labels = np.concatenate(all_labels, axis=0)
        if len(self.data.shape) == 3:
            self.data = self.data[:, None, :, :]
        elif len(self.data.shape) == 2:
            height = int(np.sqrt(self.data.shape[1]))
            self.data = self.data.reshape(-1, 1, height, height)
        if self.data.max() > 1.0:
            self.data = self.data / 255.0
        self.data = 2 * self.data - 1
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), self.labels[idx]


# --- Patch: Setup sys.path and import get_space as in adversarial_viz_topoAE.py ---
import sys
from pathlib import Path
# Determine project root (parent of scripts/)
src_path = Path(__file__).resolve().parent.parent  # project root
if not (src_path / 'src').exists():
    raise ImportError(f"Could not find src directory at {src_path / 'src'}")
# Add project root to Python path so "src" is a package
sys.path.insert(0, str(src_path))
try:
    from src.evaluation.utils import get_space
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure the src directory is in your Python path")
    sys.exit(1)

def extract_latents_and_reconstructions(
    model_path,
    data_dir,
    output_dir,
    attack_type=None,
    batch_size=126,
    device='cpu'
):
    print(f"Loading pre-trained Autoencoder...")
    model = Autoencoder()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    if device == 'cuda':
        model = model.cuda()
    print(f"Creating dataset from {data_dir}...")
    dataset = AdversarialMNISTDataset(data_dir, attack_type)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    print("Extracting latent representations manually...")
    all_latents = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels_batch in dataloader:
            if device == 'cuda':
                images = images.cuda()
            latents = model.encode(images)
            all_latents.append(latents.detach().cpu().numpy())
            all_labels.append(labels_batch)
    latent = np.concatenate(all_latents, axis=0)
    labels = np.concatenate(all_labels, axis=0)
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
    os.makedirs(output_dir, exist_ok=True)
    if attack_type:
        base_name = f"adversarial_mnist_{attack_type.replace(' ', '_')}"
    else:
        base_name = "adversarial_mnist_all"
    npz_path = os.path.join(output_dir, f"{base_name}_complete.npz")
    np.savez(
        npz_path,
        latents=latent,
        labels=labels,
        original_images=original_images,
        reconstructed_images=reconstructed_images
    )
    print(f"Saved complete data to {npz_path}")
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        print("Visualizing latent space with t-SNE...")
        latent_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(latent)
        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10', s=5)
        plt.colorbar(scatter)
        plt.title(f"t-SNE Latent Space Visualization ({attack_type or 'all'})")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plot_path = os.path.join(output_dir, f"{base_name}_latent_tsne.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved latent visualization to {plot_path}")
        from sklearn.decomposition import PCA
        print("Visualizing latent space with PCA...")
        latent_pca = PCA(n_components=2).fit_transform(latent)
        plt.figure(figsize=(8, 8))
        scatter_pca = plt.scatter(latent_pca[:, 0], latent_pca[:, 1], c=labels, cmap='tab10', s=5)
        plt.colorbar(scatter_pca)
        plt.title(f"PCA Latent Space Visualization ({attack_type or 'all'})")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        pca_plot_path = os.path.join(output_dir, f"{base_name}_latent_pca.png")
        plt.savefig(pca_plot_path, dpi=300)
        plt.close()
        print(f"Saved PCA latent visualization to {pca_plot_path}")
        plot_path = os.path.join(output_dir, f"Changed_{base_name}_latent_pca_.png")
        visualize_latents(latent, labels, plot_path)
    except ImportError:
        print("scikit-learn or matplotlib not installed, skipping latent visualization.")
    except Exception as e:
        print(f"Latent visualization failed: {e}")
    return latent, labels, original_images, reconstructed_images

def process_all_attacks(
    model_path,
    base_data_dir,
    output_dir="adversarial_mnist_results",
    device='cpu'
):
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    if not os.path.exists(base_data_dir):
        print(f"Error: Data directory not found: {base_data_dir}")
        return
    attack_dirs = [d for d in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, d))]
    print(f"Available attack types: {attack_dirs}")
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

def process_clean_mnist(
    model_path,
    output_dir="adversarial_mnist_results",
    batch_size=126,
    device='cpu'
):
    print("Processing clean MNIST test set...")
    try:
        import torchvision
        from torchvision import transforms
    except ImportError as e:
        print("torchvision not installed. Cannot process clean MNIST.")
        return
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)
    ])
    mnist_test = torchvision.datasets.MNIST(
        root="./mnist_data",
        train=False,
        download=True,
        transform=transform
    )
    dataloader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, drop_last=False)
    print(f"Loading pre-trained simple Autoencoder from {model_path}...")
    model = Autoencoder()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    if device == 'cuda':
        model = model.cuda()
    print("Extracting latent representations and reconstructions for clean MNIST...")
    all_latents = []
    all_labels = []
    all_original = []
    all_reconstructions = []
    with torch.no_grad():
        for images, labels in dataloader:
            if device == 'cuda':
                images = images.cuda()
            latents = model.encode(images)
            reconst = model.decode(latents)
            all_latents.append(latents.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            all_original.append(images.detach().cpu().numpy())
            all_reconstructions.append(reconst.detach().cpu().numpy())
    latent = np.concatenate(all_latents, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    original_images = np.concatenate(all_original, axis=0)
    reconstructed_images = np.concatenate(all_reconstructions, axis=0)
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, "mnist_clean_complete.npz")
    np.savez(
        npz_path,
        latents=latent,
        labels=labels,
        original_images=original_images,
        reconstructed_images=reconstructed_images
    )
    print(f"Saved clean MNIST complete data to {npz_path}")
    pca_plot_path = os.path.join(output_dir, "mnist_clean_latent_pca.png")
    visualize_latents(latent, labels, pca_plot_path)
    print(f"Saved clean MNIST PCA latent visualization to {pca_plot_path}")

if __name__ == "__main__":
    model_path = "clean_autoencoder.pth"
    base_data_dir = "/kaggle/input/purification/medmnist/mnist"  # or your actual data directory path
    output_dir = "./new"

    # If model file doesn't exist, train Autoencoder first
    if not os.path.exists(model_path):
        print("No trained Autoencoder found. Training on clean MNIST...")
        from torchvision import datasets, transforms
        from torch import optim
        from torch.utils.data import random_split
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)
        ])
        full_train_data = datasets.MNIST(root='./mnist_data', train=True, download=True, transform=transform)
        # Split into train and validation (90% train, 10% val)
        n_total = len(full_train_data)
        n_val = int(0.1 * n_total)
        n_train = n_total - n_val
        train_data, val_data = random_split(full_train_data, [n_train, n_val], generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = Autoencoder().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        num_epochs = 100
        patience = 10
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None
        for epoch in range(num_epochs):
            model.train()
            train_losses = []
            for imgs, _ in train_loader:
                imgs = imgs.to(device)
                recon = model(imgs)
                loss = criterion(recon, imgs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            train_loss = np.mean(train_losses)
            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for imgs, _ in val_loader:
                    imgs = imgs.to(device)
                    recon = model(imgs)
                    val_loss = criterion(recon, imgs)
                    val_losses.append(val_loss.item())
            val_loss_mean = np.mean(val_losses)
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss_mean:.4f}")
            # Early stopping
            if val_loss_mean < best_val_loss - 1e-6:
                best_val_loss = val_loss_mean
                best_model_state = model.state_dict()
                epochs_no_improve = 0
                torch.save(best_model_state, model_path)
                print(f"  (Best model so far, saved to {model_path})")
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
        # If not already saved, save final model
        if best_model_state is not None and not os.path.exists(model_path):
            torch.save(best_model_state, model_path)
        print(f"Best Autoencoder saved to {model_path}")

    process_clean_mnist(model_path, output_dir)
    process_all_attacks(model_path, base_data_dir, output_dir)