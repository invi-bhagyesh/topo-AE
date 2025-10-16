import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torchattacks
from combined import FullTopoPipeline
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, parent_dir)
from reformer import LatentReformer, LatentNet, MNIST_CNN
from src.models.approx_based import TopologicallyRegularizedAutoencoder
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class PipelineWrapper(nn.Module):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline

    def forward(self, x):
        return self.pipeline(x)[1]  # logits only

def generate_adversarial_dataset(
    pipeline,
    dataloader,
    attack_type='pgd',
    eps=0.3,
    output_path='adversarial_dataset.npz',
    device='cuda',
    **attack_kwargs
):
    model = PipelineWrapper(pipeline).to(device)
    model.eval()

    # Initialize torchattacks
    if attack_type.lower() == 'pgd':
        attacker = torchattacks.PGD(model, eps=eps, alpha=attack_kwargs.get('alpha', 2/255), steps=attack_kwargs.get('steps', 40))
    elif attack_type.lower() == 'fgsm':
        attacker = torchattacks.FGSM(model, eps=eps)
    elif attack_type.lower() == 'apgd':
        attacker = torchattacks.APGD(model, eps=eps, steps=attack_kwargs.get('steps', 40))
    elif attack_type.lower() == 'cw':
        attacker = torchattacks.CW(model, c=attack_kwargs.get('c', 1e-4), steps=attack_kwargs.get('steps', 100))
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

    all_clean, all_adv, all_labels = [], [], []
    correct_clean, correct_adv, total = 0, 0, 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Attacking batches")):
        if len(batch) == 4:
            clean_img, _, label, _ = batch
        else:
            clean_img, label = batch[:2]

        clean_img, label = clean_img.to(device), label.to(device)

        # Clean accuracy
        with torch.no_grad():
            logits_clean = model(clean_img)
            pred_clean = torch.argmax(logits_clean, dim=1)
            correct_clean += (pred_clean == label).sum().item()

        # Generate adversarial examples
        x_adv = attacker(clean_img, label)

        # Adversarial accuracy
        with torch.no_grad():
            logits_adv = model(x_adv)
            pred_adv = torch.argmax(logits_adv, dim=1)
            correct_adv += (pred_adv == label).sum().item()

        total += clean_img.size(0)
        all_clean.append(clean_img.cpu().numpy())
        all_adv.append(x_adv.cpu().numpy())
        all_labels.append(label.cpu().numpy())

    all_clean = np.concatenate(all_clean, axis=0)
    all_adv = np.concatenate(all_adv, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    clean_accuracy = 100.0 * correct_clean / total
    adv_accuracy = 100.0 * correct_adv / total
    attack_success_rate = 100.0 - adv_accuracy

    print(f"Clean accuracy: {clean_accuracy:.2f}%")
    print(f"Adversarial accuracy: {adv_accuracy:.2f}%")
    print(f"Attack success rate: {attack_success_rate:.2f}%")

    np.savez_compressed(
        output_path,
        clean_images=all_clean,
        adversarial_images=all_adv,
        labels=all_labels,
        eps=eps,
        attack_type=attack_type,
        clean_accuracy=clean_accuracy,
        adversarial_accuracy=adv_accuracy,
        attack_success_rate=attack_success_rate
    )

    return {
        'clean_images': all_clean,
        'adversarial_images': all_adv,
        'labels': all_labels,
        'clean_accuracy': clean_accuracy,
        'adversarial_accuracy': adv_accuracy,
        'attack_success_rate': attack_success_rate
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate adversarial dataset with multiple attacks")
    parser.add_argument("--attack", type=str, default="apgd", help="Attack type: apgd, pgd, autoattack, fgsm, cw, etc.")
    parser.add_argument("--eps", type=float, default=0.3, help="Perturbation budget (Linf or L2 depending on attack)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "EMNIST"], help="Dataset to use for examples")
    parser.add_argument("--batch-size", type=int, default=64, help="Dataloader batch size")
    parser.add_argument("--output", type=str, default="adversarial_dataset.npz", help="Output .npz file path")
    parser.add_argument("--norm", type=str, default="Linf", help="Norm for AutoAttack fallback ('Linf' or 'L2')")
    parser.add_argument("--version", type=str, default="standard", help="AutoAttack version: standard, plus, rand")
    parser.add_argument("--n-iter", type=int, default=100, help="Number of attack iterations (where applicable)")
    parser.add_argument("--n-restarts", type=int, default=1, help="Number of random restarts (where applicable)")
    parser.add_argument("--alpha", type=float, default=None, help="Step size for attacks that use alpha (optional)")
    parser.add_argument("--batch-save-freq", type=int, default=10, help="How often to print progress")
    parser.add_argument("--full_pipeline_path", type=str, default="./models/models--invi-bhagyesh--topo_combined/snapshots/79cb10032ff3b0c719a6a510a0c44162c564efed/MNIST_full_pipeline.pth", help="Path to saved full pipeline state_dict (.pth)")
    args = parser.parse_args()

    device = torch.device(args.device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if args.dataset == "MNIST":
        dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif args.dataset == "EMNIST":
        dataset = datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
     # Combine models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = args.dataset


    # model_path = f'./models/{dataset_name}_topo_model.pth'

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
    latent_reformer = LatentReformer()
    latent_nn = LatentNet()
    classifier = MNIST_CNN()

    full_pipeline = FullTopoPipeline(
        topo_model=topo_model,
        latent_reformer=latent_reformer,
        latent_nn=latent_nn,
        classifier=classifier,
        device=device
    )
    full_pipeline.load_state_dict(torch.load(args.full_pipeline_path, map_location=device))
    full_pipeline.to(device)
    full_pipeline.eval()

    result = generate_adversarial_dataset(
        pipeline=full_pipeline,
        dataloader=dataloader,
        attack_type=args.attack,
        eps=args.eps,
        output_path=args.output,
        device=device,
        batch_save_freq=args.batch_save_freq,
        norm=args.norm,
        version=args.version,
        n_iter=args.n_iter,
        n_restarts=args.n_restarts,
        alpha=args.alpha
    )

    print("Finished. Summary:")
    print(f"Saved to: {args.output}")
    print(f"Attack: {args.attack}, eps: {args.eps}, dataset: {args.dataset}")
    print(f"Clean accuracy: {result['clean_accuracy']:.2f}%")
    print(f"Adversarial accuracy: {result['adversarial_accuracy']:.2f}%")
    print(f"Attack success rate: {result['attack_success_rate']:.2f}%")