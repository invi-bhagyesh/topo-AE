import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torchattacks
from .combined import FullTopoPipeline
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, parent_dir)
from .reformer import LatentReformer, LatentNet, MNIST_CNN
from src.models.approx_based import TopologicallyRegularizedAutoencoder
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class PipelineWrapper(nn.Module):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline

    def forward(self, x):
        return self.pipeline(x)[1]  # logits only


class BPDAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pipeline, fallback_mode="spatial"):
        ctx.save_for_backward(x)
        ctx.pipeline = pipeline
        ctx.fallback_mode = fallback_mode
        with torch.no_grad():
            out = pipeline(x)[1]
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        pipeline = ctx.pipeline
        fallback_mode = getattr(ctx, "fallback_mode", "spatial")


        # 1) If pipeline provides a differentiable surrogate, use it (best)
        if hasattr(pipeline, "surrogate") and pipeline.surrogate is not None:
            surrogate_out = pipeline.surrogate(x)  # logits-like, differentiable
            # compute gradient of surrogate logits wrt input
            grad_x = torch.autograd.grad(
                surrogate_out, x, grad_outputs=grad_output, retain_graph=False, allow_unused=True
            )[0]
            if grad_x is not None:
                return grad_x, None, None
            # if grad_x is None, fall through to fallback

        # 2) Fallbacks (choose one)
        # grad_output shape: (B, C) typically (logits)
        # We'll convert to (B, 1, 1, 1) and expand, but keep per-sample info.

        # Option B: identity-ste (preserve per-sample gradient magnitude)
        if fallback_mode == "identity":
            # combine class gradients into a single scalar per sample
            # sum is slightly stronger than mean (scale doesn't matter for PGD step, but keep sum)
            grad_scalar = grad_output.detach().sum(dim=1).view(-1, 1, 1, 1)
            grad_input = grad_scalar.expand_as(x)
            return grad_input, None, None

        # Option C: spatial-ste (preserve some spatial structure heuristically)
        # create scalar map then modulate by input's normalized deviation from mean
        if fallback_mode == "spatial":
            eps = 1e-6
            grad_scalar = grad_output.detach().sum(dim=1).view(-1, 1, 1, 1)
            # normalized input (zero mean, scaled) as a spatial mask
            x_mean = x.mean(dim=(1, 2, 3), keepdim=True)
            x_norm = (x - x_mean) / (x.abs().mean(dim=(1,2,3), keepdim=True) + eps)
            # scale and clamp to avoid explosion
            mask = x_norm.clamp(-5.0, 5.0)
            grad_input = grad_scalar * (0.5 + 0.5 * mask)  # keep positive baseline
            return grad_input, None, None

        # Default fall back to old behaviour if unknown mode
        grad_scalar = grad_output.detach().mean(dim=1).view(-1, 1, 1, 1)
        grad_input = grad_scalar.expand_as(x)
        return grad_input, None, None



class BPDAWrapper(nn.Module):
    def __init__(self, pipeline, fallback_mode="identity"):
        super().__init__()
        self.pipeline = pipeline
        self.fallback_mode = fallback_mode

    def forward(self, x):
        return BPDAFunction.apply(x, self.pipeline, self.fallback_mode)



class EOTWrapper(nn.Module):
    def __init__(self, pipeline, n_samples=10):
        super().__init__()
        self.pipeline = pipeline
        self.n_samples = n_samples

    def forward(self, x):
        # Average logits over n stochastic forward passes.
        # Ensure that the returned logits are differentiable w.r.t. the input.
        logits = None
        for _ in range(self.n_samples):

            # out = self.pipeline(x)[1]
            # # If the pipeline returned logits that are detached (no grad),
            # # prefer a provided differentiable surrogate. If none exists,
            # # fall back to BPDAFunction.apply so backward uses the straight-through surrogate.
            # if not getattr(out, 'requires_grad', False):
            #     if hasattr(self.pipeline, 'surrogate') and self.pipeline.surrogate is not None:
            #         out = self.pipeline.surrogate(x)
            #     else:
            #         # BPDAFunction.apply will call the real pipeline in forward (no grad)
            #         # and provide surrogate/backprop behaviour in backward.
            #         out = BPDAFunction.apply(x, self.pipeline)

################# Remove for ckt
            out = self.pipeline(x)[1]
            if not out.requires_grad:
                if hasattr(self.pipeline, 'surrogate') and self.pipeline.surrogate is not None:
                    out = self.pipeline.surrogate(x)
                else:
                    # Force differentiability by cloning input and setting requires_grad=True
                    x_ = x.clone().detach().requires_grad_(True)
                    out = self.pipeline(x_)[1]  
##################

            if logits is None:
                logits = out
            else:
                logits = logits + out
        return logits / float(self.n_samples)



class BPDA_EOT_Wrapper(nn.Module):
    def __init__(self, pipeline, n_samples=10):
        super().__init__()
        self.pipeline = pipeline
        self.n_samples = n_samples

    def forward(self, x):
        # For each sample, run BPDAFunction forward (which itself calls pipeline under no_grad)
        logits = None
        for _ in range(self.n_samples):
            out = BPDAFunction.apply(x, self.pipeline)
            if logits is None:
                logits = out
            else:
                logits = logits + out
        return logits / float(self.n_samples)


# --- Inserted: ReparamWrapper ---
class ReparamWrapper(nn.Module):
    """Treat inputs as latent `z` tensors.

    This wrapper maps z -> x using a differentiable reparameterization/decoder
    provided by the pipeline. The wrapper then returns logits for the decoded
    image. The wrapper looks for a decoder in this order:
      1. pipeline.reparameterize
      2. pipeline.reparameterize_from_latent
      3. pipeline.decode_from_latent
      4. pipeline.topo_model.decode

    If none exists the wrapper raises an AttributeError.
    """

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline

        # resolve decode function
        self._decode_fn = None
        if hasattr(pipeline, 'reparameterize'):
            self._decode_fn = pipeline.reparameterize
        elif hasattr(pipeline, 'reparameterize_from_latent'):
            self._decode_fn = pipeline.reparameterize_from_latent
        elif hasattr(pipeline, 'decode_from_latent'):
            self._decode_fn = pipeline.decode_from_latent
        elif hasattr(pipeline, 'topo_model') and hasattr(pipeline.topo_model, 'decode'):
            self._decode_fn = pipeline.topo_model.decode

        if self._decode_fn is None:
            raise AttributeError('Pipeline does not expose a reparameterization/decoder required for reparam attacks')

    def forward(self, z):
        # z is a latent tensor. Map to image via decode fn then run pipeline to get logits.
        x = self._decode_fn(z)

        # prefer a differentiable surrogate when available
        if hasattr(self.pipeline, 'surrogate') and self.pipeline.surrogate is not None:
            return self.pipeline.surrogate(x)

        # If pipeline returns logits with no grad, we still want a path from z to logits
        # so call pipeline(x) without torch.no_grad so gradients flow through decode_fn.
        out = self.pipeline(x)[1]
        return out


# --- Inserted: RandomizedSmoothingWrapper ---
class RandomizedSmoothingWrapper(nn.Module):
    """Inference-time randomized smoothing wrapper.

    Averages logits over `n_samples` noisy copies of the input.
    Works for both image-space and latent-space models because it simply
    adds Gaussian noise to the tensor argument passed to `forward`.
    """
    def __init__(self, model, sigma=0.25, n_samples=20, reparam_mode=False):
        super().__init__()
        self.model = model
        self.sigma = float(sigma)
        self.n_samples = int(n_samples)
        self.reparam_mode = bool(reparam_mode)

    def forward(self, inp):
        logits = None
        for _ in range(self.n_samples):
            noise = torch.randn_like(inp) * self.sigma
            pert = inp + noise
            out = self.model(pert)
            if logits is None:
                logits = out
            else:
                logits = logits + out
        return logits / float(self.n_samples)

def generate_adversarial_dataset(
    pipeline,
    dataloader,
    attack_type='eot',
    eps=0.3,
    output_path='adversarial_dataset.npz',
    device='cuda',
    **attack_kwargs
):
    # choose wrapper for BPDA/EOT/Reparameterization if requested in attack_type or attack_kwargs
    atk_lower = attack_type.lower()
    reparam_mode = False
    if 'bpda_eot' in atk_lower or ('bpda' in atk_lower and 'eot' in atk_lower):
        print("Using BPDA + EOT wrapper for the pipeline.")
        model = BPDA_EOT_Wrapper(pipeline, n_samples=attack_kwargs.get('eot_samples', 10)).to(device)
    elif 'bpda' in atk_lower and 'reparam' not in atk_lower:
        print("Using BPDA wrapper for the pipeline.")
        model = BPDAWrapper(pipeline).to(device)
    elif 'eot' in atk_lower and 'reparam' not in atk_lower:
        print("Using EOT wrapper for the pipeline.")
        model = EOTWrapper(pipeline, n_samples=attack_kwargs.get('eot_samples', 10)).to(device)
    elif 'reparam' in atk_lower:
        print("Using Reparameterization wrapper for the pipeline. Attacks will be performed in latent space.")
        reparam_mode = True
        model = ReparamWrapper(pipeline).to(device)
    else:
        model = PipelineWrapper(pipeline).to(device)

# Optional: inference-time randomized smoothing
    if attack_kwargs.get('smoothing', True):
        sigma = attack_kwargs.get('smoothing_sigma', 0.25)
        samples = attack_kwargs.get('smoothing_samples', 10)
        model = RandomizedSmoothingWrapper(model, sigma=sigma, n_samples=samples, reparam_mode=reparam_mode).to(device)
        print(f"Enabled inference-time randomized smoothing: sigma={sigma}, samples={samples}")

    # For EOT or reparameterization you need stochastic behavior enabled in the pipeline.
    if 'eot' in atk_lower or reparam_mode:
        print("Setting pipeline to train mode for EOT or reparameterization.")
        pipeline.train()
        model.train()
    else:
        model.eval()

    # Initialize torchattacks
    # Determine base attack when wrappers like 'eot' or 'bpda' are present.
    # Accept forms like: 'pgd', 'eot_pgd', 'bpda_eot_pgd', 'apgd', 'fgsm', 'cw', 'autoattack'
    atk_lower = attack_type.lower()
    # Allow explicit override via attack_kwargs['base_attack']
    base_attack = attack_kwargs.get('base_attack')
    if base_attack is None:
        parts = atk_lower.split('_')
        # remove wrapper tokens if present
        parts = [p for p in parts if p not in ('eot', 'bpda')]
        base_attack = parts[-1] if len(parts) > 0 else 'pgd'
    base_attack = base_attack.lower()

    if base_attack == 'pgd':
        print(f"Using PGD attack (base for '{attack_type}').")
        attacker = torchattacks.PGD(model, eps=eps, alpha=attack_kwargs.get('alpha', 2/255), steps=attack_kwargs.get('steps', 40))
    elif base_attack == 'fgsm':
        print(f"Using FGSM attack (base for '{attack_type}').")
        attacker = torchattacks.FGSM(model, eps=eps)
    elif base_attack in ('apgd', 'apgd_dlr'):
        print(f"Using APGD attack (base for '{attack_type}').")
        attacker = torchattacks.APGD(model, eps=eps, steps=attack_kwargs.get('steps', 40))
    elif base_attack == 'cw':
        print(f"Using CW attack (base for '{attack_type}').")
        attacker = torchattacks.CW(model, c=attack_kwargs.get('c', 1e-4), steps=attack_kwargs.get('steps', 10))
    elif base_attack == 'spsa':
        print(f"Using SPSA attack (base for '{attack_type}').")
        try:
            lr_val = attack_kwargs.get('alpha')
            if lr_val is None:
                lr_val = 2/255

            attacker = torchattacks.SPSA(
                model,
                eps=eps,
                nb_iter=attack_kwargs.get('steps', 128),
                nb_sample=attack_kwargs.get('spsa_samples', 128),
                delta=attack_kwargs.get('delta', 0.01),
                lr=lr_val,
                max_batch_size=attack_kwargs.get('max_batch_size', 64)
            )
        except Exception as e:
            print(f"SPSA construction failed ({e}), falling back to PGD.")
            attacker = torchattacks.PGD(model, eps=eps, alpha=attack_kwargs.get('alpha', 2/255), steps=attack_kwargs.get('steps', 40))
        
    elif base_attack == 'autoattack':
        print(f"Using AutoAttack (base for '{attack_type}').")
        # AutoAttack signature varies across versions. Try to construct with common kwargs and fallback.
        try:
            attacker = torchattacks.AutoAttack(model, norm=attack_kwargs.get('norm', 'Linf'), eps=eps, version=attack_kwargs.get('version', 'standard'))
        except Exception as e:
            print(f"AutoAttack construction failed ({e}), falling back to PGD.")
            attacker = torchattacks.PGD(model, eps=eps, alpha=attack_kwargs.get('alpha', 2/255), steps=attack_kwargs.get('steps', 40))
    else:
        raise ValueError(f"Unknown base attack: {base_attack} parsed from attack_type='{attack_type}'. Provide a supported base attack or pass attack_kwargs['base_attack'].")

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
            if reparam_mode:
                # compute latent for clean image and get logits via model which expects latent
                if hasattr(pipeline, 'encode'):
                    z_clean = pipeline.encode(clean_img)
                elif hasattr(pipeline, 'encode_latent'):
                    z_clean = pipeline.encode_latent(clean_img)
                elif hasattr(pipeline, 'topo_model') and hasattr(pipeline.topo_model, 'encode'):
                    z_clean = pipeline.topo_model.encode(clean_img)
                else:
                    raise AttributeError('Pipeline does not expose an encoder required for reparam attacks')
                z_clean = z_clean.to(device)
                logits_clean = model(z_clean)
            else:
                logits_clean = model(clean_img)
            pred_clean = torch.argmax(logits_clean, dim=1)
            correct_clean += (pred_clean == label).sum().item()

        # Generate adversarial examples
        if reparam_mode:
            # compute initial latent z from clean image
            # try pipeline.encode, then pipeline.encode_latent, then pipeline.topo_model.encode
            if hasattr(pipeline, 'encode'):
                z_init = pipeline.encode(clean_img)
            elif hasattr(pipeline, 'encode_latent'):
                z_init = pipeline.encode_latent(clean_img)
            elif hasattr(pipeline, 'topo_model') and hasattr(pipeline.topo_model, 'encode'):
                z_init = pipeline.topo_model.encode(clean_img)
            else:
                raise AttributeError('Pipeline does not expose an encoder required for reparam attacks')

            z_init = z_init.detach().to(device)
            z_init.requires_grad_(True)

            # Run attacker in latent space. The attacker will perturb z tensors.
            z_adv = attacker(z_init, label)

            # Ensure attacker did not alter tensor shape. If it did, try to restore it.
            if z_adv.shape != z_init.shape:
                try:
                    z_adv = z_adv.view(z_init.shape)
                except Exception:
                    if z_adv.numel() == z_init.numel():
                        z_adv = z_adv.view(z_init.shape)
                    else:
                        raise RuntimeError(f"Attacker returned shape {z_adv.shape} but expected {z_init.shape}")

            z_adv = z_adv.to(device)

            # map back to image space using the same resolution logic as ReparamWrapper
            if hasattr(pipeline, 'reparameterize'):
                x_adv = pipeline.reparameterize(z_adv)
            elif hasattr(pipeline, 'reparameterize_from_latent'):
                x_adv = pipeline.reparameterize_from_latent(z_adv)
            elif hasattr(pipeline, 'decode_from_latent'):
                x_adv = pipeline.decode_from_latent(z_adv)
            elif hasattr(pipeline, 'topo_model') and hasattr(pipeline.topo_model, 'decode'):
                x_adv = pipeline.topo_model.decode(z_adv)
            else:
                raise AttributeError('Pipeline does not expose a decoder required for reparam attacks')

            # Adversarial accuracy on decoded images
            with torch.no_grad():
                # get logits for decoded images. prefer a surrogate if available on the pipeline
                if hasattr(pipeline, 'surrogate') and pipeline.surrogate is not None:
                    logits_adv = pipeline.surrogate(x_adv)
                else:
                    logits_adv = pipeline(x_adv)[1]
                pred_adv = torch.argmax(logits_adv, dim=1)
                correct_adv += (pred_adv == label).sum().item()

        else:
            x_adv = attacker(clean_img, label)

            # Adversarial accuracy
            with torch.no_grad():
                logits_adv = model(x_adv)
                pred_adv = torch.argmax(logits_adv, dim=1)
                correct_adv += (pred_adv == label).sum().item()

        total += clean_img.size(0)
        all_clean.append(clean_img.cpu().numpy())
        all_adv.append(x_adv.detach().cpu().numpy())
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
    parser.add_argument("--attack", type=str, default="bpda_eot", help="Attack type: apgd, pgd, autoattack, fgsm, cw, etc.")
    parser.add_argument("--eps", type=float, default=8/255, help="Perturbation budget (Linf or L2 depending on attack)")
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
    parser.add_argument("--full_pipeline_path", type=str, default="/kaggle/input/reformer_topo/pytorch/default/2/MNIST_full_pipeline.pth", help="Path to saved full pipeline state_dict (.pth)")
    parser.add_argument("--smoothing", action="store_true", default=False, help="Enable inference-time randomized smoothing (default: True)")
    parser.add_argument("--smoothing-sigma", type=float, default=0.15, help="Noise standard deviation for smoothing")
    parser.add_argument("--smoothing-samples", type=int, default=10, help="Number of noisy samples for smoothing")
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
    # latent_reformer = LatentReformer()
    # latent_nn = LatentNet()
    classifier = MNIST_CNN()

    full_pipeline = FullTopoPipeline(
        topo_model=topo_model,
        # latent_reformer=latent_reformer,
        # latent_nn=latent_nn,
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
        alpha=args.alpha,
        smoothing=args.smoothing,
        smoothing_sigma=args.smoothing_sigma,
        smoothing_samples=args.smoothing_samples,
    )

    print("Finished. Summary:")
    print(f"Saved to: {args.output}")
    print(f"Attack: {args.attack}, eps: {args.eps}, dataset: {args.dataset}")
    print(f"Clean accuracy: {result['clean_accuracy']:.2f}%")
    print(f"Adversarial accuracy: {result['adversarial_accuracy']:.2f}%")
    print(f"Attack success rate: {result['attack_success_rate']:.2f}%") 