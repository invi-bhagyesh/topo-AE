import argparse
import torch
import torch.nn as nn
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, parent_dir)
from reformer import LatentReformer, LatentNet, MNIST_CNN
from src.models.approx_based import TopologicallyRegularizedAutoencoder

parser = argparse.ArgumentParser(description='Run the full topo pipeline.')
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR', 'SYN', 'EMNIST'], help='Dataset name')
parser.add_argument('--topo_model_path', type=str, help='Path to the topo model checkpoint')
parser.add_argument('--latent_reformer_path', type=str, help='Path to the latent reformer checkpoint')
parser.add_argument('--latent_nn_path', type=str, help='Path to the latent NN checkpoint')
parser.add_argument('--classifier_path', type=str, help='Path to the classifier checkpoint')
args = parser.parse_args()

class FullTopoPipeline(nn.Module):
    def __init__(self, topo_model, latent_reformer, latent_nn, classifier, device='cpu'):
        super().__init__()
        self.topo_model = topo_model
        self.latent_reformer = latent_reformer
        self.latent_nn = latent_nn
        self.classifier = classifier
        self.device = device

    def forward(self, x):
        latent = self.topo_model.encode(x)
        topo_img = self.topo_model.decode(latent)
        latent_out = self.latent_nn(latent)
        # Step 3: latent reformer reconstruction
        recon_img, mu, logvar = self.latent_reformer(topo_img, latent_out)
        # Step 4: classification
        logits = self.classifier(recon_img)
        return recon_img, logits, mu, logvar, topo_img


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = args.dataset

    if args.topo_model_path is not None:
        model_path = args.topo_model_path
    else:
        model_path = f'./models/{dataset_name}_topo_model.pth'

    if dataset_name == 'MNIST':
        topo_model = TopologicallyRegularizedAutoencoder(
            autoencoder_model='DeepAE',
            lam=0.5002972000959738,
            toposig_kwargs={'match_edges': 'symmetric'}
        )
    elif dataset_name == 'CIFAR':
        topo_model = TopologicallyRegularizedAutoencoder(
            ae_kwargs={'input_dims': [3, 32, 32]},
            autoencoder_model='DeepAE',
            lam=1.6280214927932581,
            toposig_kwargs={'match_edges': 'symmetric'}
        )
    elif dataset_name == 'SYN':
        topo_model = TopologicallyRegularizedAutoencoder(
            ae_kwargs={'input_dims': [3, 28, 44]},
            autoencoder_model='DeepAE',
            lam=1.6280214927932581,
            toposig_kwargs={'match_edges': 'symmetric'}
        )
    elif dataset_name == 'EMNIST':
        topo_model = TopologicallyRegularizedAutoencoder(
            autoencoder_model='DeepAE',
            lam=0.5002972000959738,
            toposig_kwargs={'match_edges': 'symmetric'}
        )

    # state_dict = torch.load(model_path, map_location=device)
    # topo_model.load_state_dict(state_dict)
    # topo_model.eval()

    # latent_reformer_path = args.latent_reformer_path if args.latent_reformer_path is not None else f'./models/{dataset_name}_latent_reformer.pth'
    latent_reformer = LatentReformer()
    # latent_reformer.load_state_dict(torch.load(latent_reformer_path, map_location=device))

    # latent_nn_path = args.latent_nn_path if args.latent_nn_path is not None else f'./models/{dataset_name}_latent_nn.pth'
    latent_nn = LatentNet()
    # latent_nn.load_state_dict(torch.load(latent_nn_path, map_location=device))

    # classifier_path = args.classifier_path if args.classifier_path is not None else f'./models/{dataset_name}_classifier.pth'
    classifier = MNIST_CNN()
    # classifier.load_state_dict(torch.load(classifier_path, map_location=device))


    # Combine models
    full_pipeline = FullTopoPipeline(
        topo_model=topo_model,
        latent_reformer=latent_reformer,
        latent_nn=latent_nn,
        classifier=classifier,
        device=device
    )
    full_pipeline.load_state_dict(torch.load("./models/full_pipeline_combined.pth", map_location=device))

    full_pipeline.to(device)
    full_pipeline.eval()
    with torch.no_grad():
        # MNIST images: batch_size=2, channels=1, height=28, width=28
        dummy_input = torch.randn(2, 1, 28, 28).to(device)
        recon_img, logits, mu, logvar, _ = full_pipeline(dummy_input)

    print("Sanity Check:")
    print("Input shape:", dummy_input.shape)
    print("Reconstructed image shape:", recon_img.shape)
    print("Logits shape:", logits.shape)
    print("Latent mu shape:", mu.shape)
    print("Latent logvar shape:", logvar.shape)

    # Evaluate on MNIST test set
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    # Load MNIST test set
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    clean_correct, topo_correct, recon_correct = 0, 0, 0
    total = 0

    full_pipeline.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward through pipeline
            recon_img, logits, mu, logvar, topo_img = full_pipeline(images)

            # 1. Clean accuracy
            clean_logits = full_pipeline.classifier(images)
            clean_pred = clean_logits.argmax(dim=1)
            clean_correct += (clean_pred == labels).sum().item()

            # 2. Topo accuracy
            topo_logits = full_pipeline.classifier(topo_img.detach())
            topo_pred = topo_logits.argmax(dim=1)
            topo_correct += (topo_pred == labels).sum().item()

            # 3. Reconstructed accuracy
            recon_pred = logits.argmax(dim=1)
            recon_correct += (recon_pred == labels).sum().item()

            total += labels.size(0)

    clean_acc = 100 * clean_correct / total
    topo_acc = 100 * topo_correct / total
    recon_acc = 100 * recon_correct / total

    print(f"MNIST Evaluation:")
    print(f"Clean accuracy: {clean_acc:.2f}%")
    print(f"Topo image accuracy: {topo_acc:.2f}%")
    print(f"Reconstructed image accuracy: {recon_acc:.2f}%")