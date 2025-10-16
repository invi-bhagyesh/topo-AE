import argparse
import torch
import torch.nn as nn
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
        return recon_img, logits, mu, logvar


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

    state_dict = torch.load(model_path, map_location=device)
    topo_model.load_state_dict(state_dict)
    topo_model.eval()

    latent_reformer_path = args.latent_reformer_path if args.latent_reformer_path is not None else f'./models/{dataset_name}_latent_reformer.pth'
    latent_reformer = LatentReformer()
    latent_reformer.load_state_dict(torch.load(latent_reformer_path, map_location=device))

    latent_nn_path = args.latent_nn_path if args.latent_nn_path is not None else f'./models/{dataset_name}_latent_nn.pth'
    latent_nn = LatentNet()
    latent_nn.load_state_dict(torch.load(latent_nn_path, map_location=device))

    classifier_path = args.classifier_path if args.classifier_path is not None else f'./models/{dataset_name}_classifier.pth'
    classifier = MNIST_CNN()
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))


    # Combine models
    full_pipeline = FullTopoPipeline(
        topo_model=topo_model,
        latent_reformer=latent_reformer,
        latent_nn=latent_nn,
        classifier=classifier,
        device=device
    )
    full_pipeline.to(device)