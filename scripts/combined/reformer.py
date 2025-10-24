import random
import numpy as np
import torch
import os
import torch
import torch.nn as nn
import os
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
import torch.nn.functional as F
import shutil
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")




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

    print("Combining !!")

    # Define the output directory
    output_dir = "reconstructed_originals"
    os.makedirs(output_dir, exist_ok=True)

    # Optional: Reconstruct original images from processed characters
    if character_data and metadata:
        combined_images = combine_and_save(
            character_data,
            metadata,
            char_latents,
            char_labels,
            output_dir=output_dir
        )
        
        for img_array, filename in combined_images:
            cv2.imwrite(os.path.join(output_dir, filename), img_array)

    print("Done")

    
    if True:
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
        ## This is the dataloading part

        



"""
dataloader
"""
class MNISTTopoDataset(Dataset):
    def __init__(self, clean_images, topo_images, labels, latents):
        self.clean_images = clean_images
        self.topo_images = topo_images
        self.labels = labels
        self.latents = latents

    def __len__(self):
        return len(self.clean_images)

    def __getitem__(self, idx):
        clean = torch.tensor(self.clean_images[idx], dtype=torch.float32)
        topo = torch.tensor(self.topo_images[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        latent = torch.tensor(self.latents[idx], dtype=torch.float32)
        return clean, topo, label, latent

def get_mnist_topo_loaders(npz_path, batch_size=64, val_split=0.1):
    data = np.load(npz_path)
    clean_images = data['clean']
    topo_images = data['reconstructed']
    labels = data['label']
    latents = data['latent']

    full_dataset = MNISTTopoDataset(clean_images, topo_images, labels, latents)

    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader




"""
Latent NN to map latent vector to bottleneck
"""
class LatentNet(nn.Module):
    def __init__(self, latent_dim=64, bottleneck_h=14, bottleneck_w=14):
        super().__init__()
        bottleneck_size = bottleneck_h * bottleneck_w
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, bottleneck_size)
        )
        self.bottleneck_h = bottleneck_h
        self.bottleneck_w = bottleneck_w

    def forward(self, z):
        out = self.fc(z)  # [batch_size, bottleneck_h * bottleneck_w]
        out = out.view(-1, 1, self.bottleneck_h, self.bottleneck_w)
        return out


"""
Reformer
"""
class LatentReformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.ReLU(),  # Changed from Sigmoid
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.ReLU()  # Changed from Sigmoid
        )
        
        self.fc_mu = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.fc_logvar = nn.Conv2d(1, 1, kernel_size=3, padding=1)

        # Decoder with proper activation
        self.decoder = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.ReLU(),  # Changed from Sigmoid
            nn.Upsample(scale_factor=2),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.ReLU(),  # Changed from Sigmoid
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.Tanh()  # ✅ ADDED: Outputs [-1, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, img, latent_bottleneck):
        x = self.encoder(img)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
    
        batch_size = latent_bottleneck.size(0)
        latent_flat = latent_bottleneck.view(batch_size, -1)
        latent_bottleneck_expanded = latent_flat.view(batch_size, -1, 1, 1)
        latent_bottleneck_expanded = latent_bottleneck_expanded.expand(-1, -1, z.size(2), z.size(3))
    
        z_cat = torch.cat([z, latent_bottleneck_expanded.to(z.device)], dim=1)
    
        proj = nn.Conv2d(z_cat.size(1), 2, kernel_size=1).to(z.device)
        z = proj(z_cat)
    
        x_recon = self.decoder(z)
        return x_recon, mu, logvar




import torch
import torch.nn as nn
import torch.nn.functional as F


class EMNIST_CNN(nn.Module):
    def __init__(self, num_classes=26):  
        super(EMNIST_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=3, padding=1),   # Changed: 3 -> 1 input channels (grayscale)
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),  # Second Conv
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),  # Third Conv
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                  # MaxPool 2x2
            nn.Conv2d(96, 192, kernel_size=3, padding=1), # Fourth Conv
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),# Fifth Conv
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),# Sixth Conv
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                  # MaxPool 2x2
            nn.Conv2d(192, 192, kernel_size=3, padding=1),# Seventh Conv
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1),           # Eighth Conv 1x1
            nn.ReLU(inplace=True),
            nn.Conv2d(192, num_classes, kernel_size=1),   # Ninth Conv 1x1, output channels = num_classes
        )
        # Global Average Pooling will be performed in forward()
        
    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, x.shape[2:]) # Global Average Pooling over spatial dims
        x = x.view(x.size(0), -1)
        return x
"""
Classifier
"""
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        
        # First Conv Block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Input: 1x28x28 → Output: 32x28x28
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # 32x28x28 → 32x28x28
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x28x28 → 32x14x14
        
        # Second Conv Block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 64x14x14
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # → 64x7x7
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 7 * 7, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
    
    def forward(self, x):
        # First Conv Block
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        # Second Conv Block
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Softmax applied in loss function
        return x


import torchmetrics
def train_latent_vae(latent_reformer, latent_nn, model, train_loader, val_loader,
                     epochs, lr, device, alpha=0.5, beta=2.0, warmup_epochs=10):

    latent_reformer.to(device)
    latent_nn.to(device)
    model.to(device)
    optimizer = torch.optim.Adam(
        list(latent_reformer.parameters()) ,
        lr=lr
    )
    criterion_recon = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()

    # Initialize torchmetrics PSNR and SSIM for evaluation
    psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    for epoch in range(epochs):
        latent_nn.train()
        latent_reformer.train()
        model.eval()
        train_loss = 0.0

        # Freeze encoder during warmup epochs
        if epoch < warmup_epochs:
            for param in latent_reformer.encoder.parameters():
                param.requires_grad = False
        else:
            for param in latent_reformer.encoder.parameters():
                param.requires_grad = True

        for clean_img, topo_img, label, latent_vec in train_loader:
            clean_img = clean_img.to(device)
            topo_img = topo_img.to(device)
            label = label.to(device)
            latent_vec = latent_vec.to(device)

            latent_bottleneck = latent_nn(latent_vec)
            recon_output, mu, logvar = latent_reformer(topo_img, latent_bottleneck)
            # recon_output, mu, logvar = latent_reformer(topo_img)

            # Reconstruction loss
            loss_recon = criterion_recon(recon_output, clean_img)

            # KL Divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / clean_img.size(0) / clean_img.numel()  # Normalize by batch and pixels

            with torch.no_grad():
                for param in model.parameters():
                    param.requires_grad = False
            logits_class = model(recon_output)
            loss_class = criterion_class(logits_class, label)

            # Total loss with KL term
            loss = alpha * loss_recon + beta * loss_class + kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * clean_img.size(0)

        train_loss = train_loss / len(train_loader.dataset)

        # Validation phase
        latent_reformer.eval()
        val_loss = 0.0
        psnr_metric.reset()
        ssim_metric.reset()
        total_samples = 0

        with torch.no_grad():
            for clean_img, topo_img, label, latent_vec in val_loader:
                clean_img = clean_img.to(device)
                topo_img = topo_img.to(device)
                latent_vec = latent_vec.to(device)
                label = label.to(device)

                latent_bottleneck = latent_nn(latent_vec)
                recon_output, mu, logvar = latent_reformer(topo_img, latent_bottleneck)
                

                loss_recon = criterion_recon(recon_output, clean_img)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = kl_loss / clean_img.size(0) / clean_img.numel()

                logits_class = model(recon_output)
                loss_class = criterion_class(logits_class, label)

                loss = alpha * loss_recon + beta * loss_class + kl_loss
                val_loss += loss.item() * clean_img.size(0)

                # Update PSNR and SSIM metrics
                psnr_metric.update(recon_output, clean_img)
                ssim_metric.update(recon_output, clean_img)
                total_samples += clean_img.size(0)

        val_loss /= len(val_loader.dataset)
        avg_psnr = psnr_metric.compute().item()
        avg_ssim = ssim_metric.compute().item()

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | PSNR: {avg_psnr:.3f} | SSIM: {avg_ssim:.3f}")

    print("Training Complete!")
    return latent_reformer




import matplotlib.pyplot as plt
import numpy as np
import torch
def visualize_model_reconstruction(latent_reformer, latent_nn, val_loader, device, n_samples=10):
# def visualize_model_reconstruction(latent_reformer, val_loader, device, n_samples=10):
    """
    Visualizes clean_img, topo_img, and latent_reformer output from the validation loader.
    Shows n_samples images from the first batch in val_loader.
    """
    latent_reformer.eval()
    with torch.no_grad():
        # Get one batch from val_loader
        for clean_img, topo_img, label, latent_vec in val_loader:
            clean_img = clean_img.to(device)
            topo_img = topo_img.to(device)
            latent_vec = latent_vec.to(device)

            # bottleneck
            latent_bottleneck = latent_nn(latent_vec)
            output, mu, logvar = latent_reformer(topo_img, latent_bottleneck)
            # output, mu, logvar = latent_reformer(topo_img)

            # Convert tensors to numpy
            clean_np = clean_img.detach().cpu().numpy()
            topo_np = topo_img.detach().cpu().numpy()
            output_np = output.detach().cpu().numpy()

            batch_size = clean_np.shape[0]
            n_show = min(n_samples, batch_size)

            fig, axes = plt.subplots(n_show, 3, figsize=(9, 2 * n_show))
            for i in range(n_show):
                c_img = clean_np[i]
                t_img = topo_np[i]
                o_img = output_np[i]

                # Handle grayscale (channel-first [1, H, W] → [H, W])
                if c_img.ndim == 3 and c_img.shape[0] == 1:
                    c_img = c_img.squeeze(0)
                if t_img.ndim == 3 and t_img.shape[0] == 1:
                    t_img = t_img.squeeze(0)
                if o_img.ndim == 3 and o_img.shape[0] == 1:
                    o_img = o_img.squeeze(0)

                axes[i, 0].imshow(c_img, cmap='gray')
                axes[i, 0].set_title('Clean')
                axes[i, 0].axis('off')

                axes[i, 1].imshow(t_img, cmap='gray')
                axes[i, 1].set_title('Topo')
                axes[i, 1].axis('off')

                axes[i, 2].imshow(o_img, cmap='gray')
                axes[i, 2].set_title('Reconstructed')
                axes[i, 2].axis('off')

            plt.tight_layout()
            plt.show()
            break  # only visualize first batch


import torch
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
)

def evaluate_metrics_topo_vs_reconstruction(
    classifier, latent_reformer, latent_nn,
    val_loader, device, num_classes=10
):
# def evaluate_metrics_topo_vs_reconstruction(
#     classifier, latent_reformer,
#     val_loader, device, num_classes=10
# ):
    classifier.eval()
    latent_reformer.eval()
    latent_nn.eval()
    
    all_labels = []
    all_preds_topo = []
    all_preds_recon = []
    all_probs_topo = []
    all_probs_recon = []
    
    with torch.no_grad():
        for clean_img, topo_img, label, latent_vec in val_loader:
            topo_img = topo_img.to(device)
            latent_vec = latent_vec.to(device)
            label = label.to(device)
            
            # --- Topo branch ---
            logits_topo = classifier(topo_img)
            preds_topo = torch.argmax(logits_topo, dim=1)
            probs_topo = torch.softmax(logits_topo, dim=1)
            
            # --- Reconstruction branch ---
            latent_bottleneck = latent_nn(latent_vec)
            recon_img, _, _ = latent_reformer(topo_img, latent_bottleneck)
            # recon_img, _, _ = latent_reformer(topo_img)
            if recon_img.ndim == 3:
                recon_img = recon_img.unsqueeze(1)
            logits_recon = classifier(recon_img)
            preds_recon = torch.argmax(logits_recon, dim=1)
            probs_recon = torch.softmax(logits_recon, dim=1)
            
            all_labels.extend(label.cpu().numpy())
            all_preds_topo.extend(preds_topo.cpu().numpy())
            all_preds_recon.extend(preds_recon.cpu().numpy())
            all_probs_topo.extend(probs_topo.cpu().numpy())
            all_probs_recon.extend(probs_recon.cpu().numpy())
    
    y_true = np.array(all_labels)
    y_pred_topo = np.array(all_preds_topo)
    y_pred_recon = np.array(all_preds_recon)
    y_prob_topo = np.array(all_probs_topo)
    y_prob_recon = np.array(all_probs_recon)
    
    # Binarize true labels for AUC computation if multi-class
    if num_classes is not None:
        y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    else:
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))

    def compute_metrics(y_true, y_pred, y_prob, y_true_bin, num_classes):
        metrics = {}
        metrics["F1"] = f1_score(y_true, y_pred, average="macro")
        metrics["Precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["Recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
        try:
            metrics["AUC"] = roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="macro")
        except Exception:
            metrics["AUC"] = float("nan")
        
        cm = confusion_matrix(y_true, y_pred)
        # Per-class TP, FP, TN, FN
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        tn = cm.sum() - (tp + fp + fn)
        
        # Aggregate global TP, FP, TN, FN by summing all classes
        metrics["TP"] = int(tp.sum())
        metrics["FP"] = int(fp.sum())
        metrics["FN"] = int(fn.sum())
        metrics["TN"] = int(tn.sum())
        
        return metrics
    topo_metrics = compute_metrics(y_true, y_pred_topo, y_prob_topo, y_true_bin,num_classes)
    recon_metrics = compute_metrics(y_true, y_pred_recon, y_prob_recon, y_true_bin,num_classes)

    print("\n--- Metrics on Topo Images ---")
    for k, v in topo_metrics.items():
        print(f"{k}: {v}")
    print("\n--- Metrics on Reconstructed Images ---")
    for k, v in recon_metrics.items():
        print(f"{k}: {v}")

    


import torch
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
)

def evaluate_metrics_classifier_only(classifier, val_loader, device, num_classes=10):
    classifier.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for _, topo_img, label, _ in val_loader:
            topo_img = topo_img.to(device)
            label = label.to(device)
            
            # Classifier directly on topo images
            logits = classifier(topo_img)
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    # Binarize labels for AUC if multi-class
    if num_classes is not None:
        y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    else:
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))

    metrics = {}
    metrics["F1"] = f1_score(y_true, y_pred, average="macro")
    metrics["Precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["Recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        metrics["AUC"] = roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="macro")
    except Exception:
        metrics["AUC"] = float("nan")
    
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    metrics["TP"] = int(tp.sum())
    metrics["FP"] = int(fp.sum())
    metrics["FN"] = int(fn.sum())
    metrics["TN"] = int(tn.sum())

    print("\n--- Metrics on Topo Images (Classifier Only) ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    
    return metrics




# All code that executes anything automatically is wrapped below
if __name__ == "__main__":
    # Set random seed
    set_seed(42)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loaders
    train_loader, val_loader = get_mnist_topo_loaders(
        "/kaggle/input/invi_gan_mnist/pytorch/default/3/mnist_gan_inference/clean/mnist_train_complete.npz"
    )

    # Model parameters
    bottleneck_h = 14
    bottleneck_w = 14
    latent_dim = 64

    # Instantiate models
    latent_reformer = LatentReformer()
    latent_nn = LatentNet(latent_dim=latent_dim, bottleneck_h=bottleneck_h, bottleneck_w=bottleneck_w)
    model = MNIST_CNN()
    model.load_state_dict(torch.load('/kaggle/input/classifiers/Pretrained_classifiers/mnist.pth', map_location='cpu'))
    model.to(device)
    model.eval()
    latent_reformer.to(device)
    latent_nn.to(device)

    # Train Latent VAE
    latent_reformer = train_latent_vae(
        latent_reformer, latent_nn, model,
        train_loader, val_loader,
        epochs=50, lr=1e-3, device=device
    )

    # Visualize reconstruction
    visualize_model_reconstruction(latent_reformer, latent_nn, val_loader, device, n_samples=10)

    # Evaluation on validation set
    evaluate_metrics_classifier_only(model, val_loader, device)
    evaluate_metrics_topo_vs_reconstruction(model, latent_reformer, latent_nn, val_loader, device)

    # Function to get adversarial MNIST loaders
    def get_advmnist_topo_loaders(npz_path, batch_size=64, val_split=0.1):
        data = np.load(npz_path)
        clean_images = data['original_images']
        topo_images = data['reconstructed_images']
        labels = data['labels']
        latents = data['latents']
        full_dataset = MNISTTopoDataset(clean_images, topo_images, labels, latents)
        full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
        return full_loader

    # Evaluation on adversarial sets
    adv_loader_strong = get_advmnist_topo_loaders(
        "/kaggle/input/invi_gan_mnist/pytorch/default/3/mnist_gan_inference/adversarial/adversarial_mnist_cw_strong_complete.npz"
    )
    evaluate_metrics_classifier_only(model, adv_loader_strong, device)

    adv_loader_weak = get_advmnist_topo_loaders(
        "/kaggle/input/invi_gan_mnist/pytorch/default/3/mnist_gan_inference/adversarial/adversarial_mnist_cw_weak_complete.npz"
    )
    evaluate_metrics_classifier_only(model, adv_loader_weak, device)

    evaluate_metrics_topo_vs_reconstruction(model, latent_reformer, latent_nn, adv_loader_weak, device)