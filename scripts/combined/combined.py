import torch
import torch.nn as nn
from reformer import LatentReformer, LatentNet, MNIST_CNN
from src.models.approx_based import TopologicallyRegularizedAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_name = 'MNIST'  

if dataset_name == 'MNIST':
    topo_model = TopologicallyRegularizedAutoencoder(
        autoencoder_model='DeepAE',
        lam=0.5002972000959738,
        toposig_kwargs={'match_edges': 'symmetric'}
    )
elif dataset_name == 'CIFAR':
    topo_model = TopologicallyRegularizedAutoencoder(
        ae_kwargs={'input_dims':[3,32,32]},
        autoencoder_model='DeepAE',
        lam=1.6280214927932581,
        toposig_kwargs={'match_edges':'symmetric'}
    )
elif dataset_name == 'SYN':
    topo_model = TopologicallyRegularizedAutoencoder(
        ae_kwargs={'input_dims':[3,28,44]},
        autoencoder_model='DeepAE',
        lam=1.6280214927932581,
        toposig_kwargs={'match_edges':'symmetric'}
    )
elif dataset_name == 'EMNIST':
    topo_model = TopologicallyRegularizedAutoencoder(
        autoencoder_model='DeepAE',
        lam=0.5002972000959738,
        toposig_kwargs={'match_edges':'symmetric'}
    )

# Other models
latent_reformer = LatentReformer()
latent_nn = LatentNet()
classifier = MNIST_CNN()


class FullTopoPipeline(nn.Module):
    def __init__(self, topo_model, latent_reformer, classifier, device='cpu'):
        super().__init__()
        self.topo_model = topo_model
        self.latent_reformer = latent_reformer
        self.classifier = classifier
        self.device = device

    def forward(self, x):

        latent = self.topo_model.encode(x)

        topo_img = self.topo_model.decode(latent)

        # Step 3: latent reformer reconstruction
        recon_img, mu, logvar = self.latent_reformer(topo_img, latent)

        # Step 4: classification
        logits = self.classifier(recon_img)
        return recon_img, logits, mu, logvar


# Combine models
full_pipeline = FullTopoPipeline(topo_model, latent_reformer, classifier, device=device)
full_pipeline.to(device)