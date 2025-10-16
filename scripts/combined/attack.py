import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
# APGD and autoattack

from torchvision import datasets, transforms

class PipelineWrapper(nn.Module):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline

    def forward(self, x):
        # FullTopoPipeline.forward -> (recon_img, logits, mu, logvar)
        return self.pipeline(x)[1]  # return logits only

class APGD:
    """
    Auto-PGD (APGD) Attack Implementation
    Based on "Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks"
    """
    def __init__(
        self,
        model,
        eps=0.3,
        n_iter=100,
        norm='Linf',
        n_restarts=1,
        loss='ce',
        device='cuda'
    ):
        """
        Args:
            model: Target model
            eps: Maximum perturbation
            n_iter: Number of iterations
            norm: 'Linf', 'L2', or 'L1'
            n_restarts: Number of random restarts
            loss: 'ce' (cross-entropy) or 'dlr' (difference of logits ratio)
            device: Device to run on
        """
        self.model = model
        self.eps = eps
        self.n_iter = n_iter
        self.norm = norm
        self.n_restarts = n_restarts
        self.loss_type = loss
        self.device = device
        
    def dlr_loss(self, logits, labels):
        """Difference of Logits Ratio loss"""
        correct_logit = logits[torch.arange(len(labels)), labels].clone()
        logits_wo_correct = logits.clone()
        logits_wo_correct[torch.arange(len(labels)), labels] = -float('inf')
        max_other_logit = logits_wo_correct.max(dim=1)[0]
        
        # Get 3rd largest for denominator
        logits_sorted = logits.sort(dim=1, descending=True)[0]
        diff = -(correct_logit - max_other_logit) / (logits_sorted[:, 0] - logits_sorted[:, 2] + 1e-12)
        return diff.sum()
    
    def ce_loss(self, logits, labels):
        """Cross-entropy loss"""
        return F.cross_entropy(logits, labels, reduction='sum')
    
    def compute_loss(self, logits, labels):
        """Compute loss based on loss_type"""
        if self.loss_type == 'dlr':
            return self.dlr_loss(logits, labels)
        else:
            return self.ce_loss(logits, labels)
    
    def project(self, x, x_adv, eps):
        """Project perturbation to norm ball"""
        if self.norm == 'Linf':
            delta = torch.clamp(x_adv - x, -eps, eps)
        elif self.norm == 'L2':
            delta = x_adv - x
            delta_norm = torch.sqrt((delta ** 2).sum([1, 2, 3], keepdim=True))
            delta = delta / torch.max(delta_norm, torch.ones_like(delta_norm) * 1e-12) * torch.min(delta_norm, torch.ones_like(delta_norm) * eps)
        elif self.norm == 'L1':
            delta = x_adv - x
            delta_flat = delta.view(delta.size(0), -1)
            delta_norm = delta_flat.abs().sum(dim=1, keepdim=True)
            delta = delta / torch.max(delta_norm.view(-1, 1, 1, 1), torch.ones_like(delta_norm).view(-1, 1, 1, 1) * 1e-12) * torch.min(delta_norm.view(-1, 1, 1, 1), torch.ones_like(delta_norm).view(-1, 1, 1, 1) * eps)
        
        x_adv = x + delta
        x_adv = torch.clamp(x_adv, 0, 1)
        return x_adv
    
    def attack_single_run(self, x, y, eps):
        """Single run of APGD"""
        x_adv = x.clone().detach()
        
        # Random initialization
        if self.norm == 'Linf':
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
        elif self.norm == 'L2':
            delta = torch.randn_like(x_adv)
            delta_norm = torch.sqrt((delta ** 2).sum([1, 2, 3], keepdim=True))
            delta = delta / delta_norm * eps * torch.rand(x_adv.size(0), 1, 1, 1, device=self.device)
            x_adv = x_adv + delta
        
        x_adv = torch.clamp(x_adv, 0, 1)
        x_adv = self.project(x, x_adv, eps)
        
        # Step size scheduling
        step_size = 2.0 * eps
        n_iter = self.n_iter
        
        # Momentum
        momentum = torch.zeros_like(x)
        
        best_loss = -float('inf')
        best_x_adv = x_adv.clone()
        
        for i in range(n_iter):
            x_adv.requires_grad = True
            
            with torch.enable_grad():
                logits = self.model(x_adv)
                loss = self.compute_loss(logits, y)
            
            grad = torch.autograd.grad(loss, x_adv)[0]
            
            # Save best adversarial example (worst loss for model)
            if loss.item() > best_loss:
                best_loss = loss.item()
                best_x_adv = x_adv.clone().detach()
            
            with torch.no_grad():
                # Momentum update
                grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1, keepdim=True).view(-1, 1, 1, 1)
                grad = grad / (grad_norm + 1e-12)
                
                # Step size scheduling (reduce step size as iterations progress)
                if i > 0 and i % (n_iter // 4) == 0:
                    step_size = step_size / 2.0
                
                # Update
                x_adv = x_adv.detach() + step_size * grad.sign()
                
                # Project back
                x_adv = self.project(x, x_adv, eps)
        
        return best_x_adv
    
    def perturb(self, x, y):
        """
        Generate adversarial examples with multiple restarts
        
        Args:
            x: Clean images [batch, channels, height, width]
            y: True labels [batch]
            
        Returns:
            x_adv: Adversarial examples
        """
        self.model.eval()
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        best_x_adv = x.clone()
        max_loss = torch.zeros(x.size(0), device=self.device) - float('inf')
        
        for restart in range(self.n_restarts):
            x_adv = self.attack_single_run(x, y, self.eps)
            
            with torch.no_grad():
                logits = self.model(x_adv)
                loss = F.cross_entropy(logits, y, reduction='none')
                
                # Keep adversarial examples with highest loss (worst for model)
                improved = loss > max_loss
                max_loss = torch.where(improved, loss, max_loss)
                best_x_adv = torch.where(improved.view(-1, 1, 1, 1), x_adv, best_x_adv)
        
        return best_x_adv


class MultiRestartPGD:
    """
    Multi-restart PGD Attack
    """
    def __init__(
        self,
        model,
        eps=0.3,
        alpha=0.01,
        n_iter=100,
        n_restarts=10,
        norm='Linf',
        random_start=True,
        device='cuda'
    ):
        """
        Args:
            model: Target model
            eps: Maximum perturbation
            alpha: Step size
            n_iter: Number of iterations per restart
            n_restarts: Number of random restarts
            norm: 'Linf' or 'L2'
            random_start: Whether to use random initialization
            device: Device to run on
        """
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.n_iter = n_iter
        self.n_restarts = n_restarts
        self.norm = norm
        self.random_start = random_start
        self.device = device
    
    def pgd_single_run(self, x, y):
        """Single PGD run"""
        x_adv = x.clone().detach()
        
        # Random initialization
        if self.random_start:
            if self.norm == 'Linf':
                x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
            elif self.norm == 'L2':
                delta = torch.randn_like(x_adv)
                delta_norm = torch.sqrt((delta ** 2).sum([1, 2, 3], keepdim=True))
                delta = delta / delta_norm * self.eps * torch.rand(x_adv.size(0), 1, 1, 1, device=self.device)
                x_adv = x_adv + delta
            x_adv = torch.clamp(x_adv, 0, 1)
        
        for i in range(self.n_iter):
            x_adv.requires_grad = True
            
            with torch.enable_grad():
                logits = self.model(x_adv)
                loss = F.cross_entropy(logits, y)
            
            grad = torch.autograd.grad(loss, x_adv)[0]
            
            with torch.no_grad():
                if self.norm == 'Linf':
                    x_adv = x_adv.detach() + self.alpha * grad.sign()
                    delta = torch.clamp(x_adv - x, -self.eps, self.eps)
                elif self.norm == 'L2':
                    grad_norm = torch.sqrt((grad ** 2).sum([1, 2, 3], keepdim=True))
                    grad = grad / (grad_norm + 1e-12)
                    x_adv = x_adv.detach() + self.alpha * grad
                    delta = x_adv - x
                    delta_norm = torch.sqrt((delta ** 2).sum([1, 2, 3], keepdim=True))
                    delta = delta / torch.max(delta_norm, torch.ones_like(delta_norm) * 1e-12) * torch.min(delta_norm, torch.ones_like(delta_norm) * self.eps)
                
                x_adv = torch.clamp(x + delta, 0, 1)
        
        return x_adv
    
    def perturb(self, x, y):
        """
        Generate adversarial examples with multiple restarts
        
        Args:
            x: Clean images [batch, channels, height, width]
            y: True labels [batch]
            
        Returns:
            x_adv: Adversarial examples
        """
        self.model.eval()
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        best_x_adv = x.clone()
        max_loss = torch.zeros(x.size(0), device=self.device) - float('inf')
        
        for restart in range(self.n_restarts):
            x_adv = self.pgd_single_run(x, y)
            
            with torch.no_grad():
                logits = self.model(x_adv)
                loss = F.cross_entropy(logits, y, reduction='none')
                
                # Keep adversarial examples with highest loss
                improved = loss > max_loss
                max_loss = torch.where(improved, loss, max_loss)
                best_x_adv = torch.where(improved.view(-1, 1, 1, 1), x_adv, best_x_adv)
        
        return best_x_adv


def run_autoattack(model, x, y, eps=0.3, norm='Linf', version='standard', device='cuda'):
    """
    Run AutoAttack on the model
    
    Args:
        model: Target model (should output logits)
        x: Clean images [batch, channels, height, width]
        y: True labels [batch]
        eps: Maximum perturbation
        norm: 'Linf' or 'L2'
        version: 'standard' or 'rand' (with additional random restarts)
        device: Device to run on
        
    Returns:
        x_adv: Adversarial examples
        robust_accuracy: Accuracy on adversarial examples
    """
    try:
        from autoattack import AutoAttack
    except ImportError:
        print("AutoAttack library not installed. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'autoattack'])
        from autoattack import AutoAttack
    
    model.eval()
    
    x = x.to(device)
    y = y.to(device)
    
    # Initialize AutoAttack
    adversary = AutoAttack(
        model, 
        norm=norm, 
        eps=eps, 
        version=version,
        device=device,
        verbose=True
    )
    
    # Run attack
    x_adv = adversary.run_standard_evaluation(x, y, bs=x.size(0))
    
    # Compute robust accuracy
    with torch.no_grad():
        logits = model(x_adv)
        preds = torch.argmax(logits, dim=1)
        robust_accuracy = (preds == y).float().mean().item()
    
    return x_adv, robust_accuracy


def generate_adversarial_dataset(
    pipeline,
    dataloader,
    attack_type='apgd',
    eps=0.3,
    output_path='adversarial_dataset.npz',
    device='cuda',
    **attack_kwargs
):
    """
    Generate adversarial examples for entire dataset
    
    Args:
        pipeline: Unified topological pipeline
        dataloader: DataLoader with clean images
        attack_type: 'apgd', 'pgd', or 'autoattack'
        eps: Maximum perturbation
        output_path: Path to save adversarial dataset
        device: Device to run on
        **attack_kwargs: Additional arguments for attack
        
    Returns:
        Dictionary with adversarial data
    """
    # Wrap pipeline for adversarial attacks
    model = PipelineWrapper(pipeline).to(device)
    model.eval()
    
    # Initialize attack
    if attack_type == 'apgd':
        attacker = APGD(
            model=model,
            eps=eps,
            n_iter=attack_kwargs.get('n_iter', 100),
            norm=attack_kwargs.get('norm', 'Linf'),
            n_restarts=attack_kwargs.get('n_restarts', 1),
            loss=attack_kwargs.get('loss', 'dlr'),
            device=device
        )
    elif attack_type == 'pgd':
        attacker = MultiRestartPGD(
            model=model,
            eps=eps,
            alpha=attack_kwargs.get('alpha', 0.01),
            n_iter=attack_kwargs.get('n_iter', 100),
            n_restarts=attack_kwargs.get('n_restarts', 10),
            norm=attack_kwargs.get('norm', 'Linf'),
            device=device
        )
    elif attack_type == 'autoattack':
        # AutoAttack will be called per batch
        attacker = None
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")
    
    all_clean = []
    all_adv = []
    all_labels = []
    all_latents = []
    
    correct_clean = 0
    correct_adv = 0
    total = 0
    
    print(f"\nGenerating {attack_type.upper()} adversarial examples...")
    print(f"Epsilon: {eps}")
    print(f"Device: {device}\n")
    
    with torch.no_grad():
        pipeline.eval()
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Attacking batches")):
        if len(batch) == 4:
            clean_img, _, label, latent = batch
        else:
            clean_img, label = batch[:2]
            latent = None
        
        clean_img = clean_img.to(device)
        label = label.to(device)
        
        # Check clean accuracy
        with torch.no_grad():
            logits_clean = model(clean_img)
            pred_clean = torch.argmax(logits_clean, dim=1)
            correct_clean += (pred_clean == label).sum().item()
        
        # Generate adversarial examples
        if attack_type == 'autoattack':
            x_adv, _ = run_autoattack(
                model, clean_img, label, 
                eps=eps, 
                norm=attack_kwargs.get('norm', 'Linf'),
                version=attack_kwargs.get('version', 'standard'),
                device=device
            )
        else:
            x_adv = attacker.perturb(clean_img, label)
        
        # Check adversarial accuracy
        with torch.no_grad():
            logits_adv = model(x_adv)
            pred_adv = torch.argmax(logits_adv, dim=1)
            correct_adv += (pred_adv == label).sum().item()
        
        total += clean_img.size(0)
        
        # Extract latents from adversarial examples
        with torch.no_grad():
            latent_adv = pipeline.extract_latent(x_adv)
            if latent_adv.dim() > 2:
                latent_adv = latent_adv.view(latent_adv.size(0), -1)
        
        # Store results
        all_clean.append(clean_img.cpu().numpy())
        all_adv.append(x_adv.cpu().numpy())
        all_labels.append(label.cpu().numpy())
        all_latents.append(latent_adv.cpu().numpy())
        
        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            clean_acc = 100.0 * correct_clean / total
            adv_acc = 100.0 * correct_adv / total
            print(f"\nBatch {batch_idx + 1}: Clean Acc: {clean_acc:.2f}% | Adv Acc: {adv_acc:.2f}%")
    
    # Concatenate all batches
    all_clean = np.concatenate(all_clean, axis=0)
    all_adv = np.concatenate(all_adv, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_latents = np.concatenate(all_latents, axis=0)
    
    # Compute final statistics
    clean_accuracy = 100.0 * correct_clean / total
    adv_accuracy = 100.0 * correct_adv / total
    attack_success_rate = 100.0 - adv_accuracy
    
    print("\n" + "="*60)
    print(f"ATTACK SUMMARY ({attack_type.upper()})")
    print("="*60)
    print(f"Total samples: {total}")
    print(f"Clean accuracy: {clean_accuracy:.2f}%")
    print(f"Adversarial accuracy: {adv_accuracy:.2f}%")
    print(f"Attack success rate: {attack_success_rate:.2f}%")
    print(f"Average perturbation (L∞): {np.abs(all_adv - all_clean).max(axis=(1,2,3)).mean():.4f}")
    print(f"Average perturbation (L2): {np.sqrt(((all_adv - all_clean)**2).sum(axis=(1,2,3))).mean():.4f}")
    print("="*60 + "\n")
    
    # Save adversarial dataset
    np.savez_compressed(
        output_path,
        clean_images=all_clean,
        adversarial_images=all_adv,
        labels=all_labels,
        latents=all_latents,
        eps=eps,
        attack_type=attack_type,
        clean_accuracy=clean_accuracy,
        adversarial_accuracy=adv_accuracy,
        attack_success_rate=attack_success_rate
    )
    
    print(f"Adversarial dataset saved to: {output_path}")
    
    return {
        'clean_images': all_clean,
        'adversarial_images': all_adv,
        'labels': all_labels,
        'latents': all_latents,
        'clean_accuracy': clean_accuracy,
        'adversarial_accuracy': adv_accuracy,
        'attack_success_rate': attack_success_rate
    }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    from combined import full_pipeline

    # Use the unified full_pipeline for MNIST
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transform for MNIST/EMNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # MNIST dataset and dataloader
    mnist_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    dataloader_mnist = DataLoader(mnist_dataset, batch_size=64, shuffle=False)

    # EMNIST dataset (letters) and dataloader
    emnist_dataset = datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)
    dataloader_emnist = DataLoader(emnist_dataset, batch_size=64, shuffle=False)

    # Wrap the existing full_pipeline for attacks
    pipeline_mnist = PipelineWrapper(full_pipeline).to(device)


    # Example 1: APGD Attack with DLR loss
    print("\n" + "="*60)
    print("Example 1: APGD Attack with DLR Loss")
    print("="*60)

    apgd_data = generate_adversarial_dataset(
        pipeline=full_pipeline,
        dataloader=dataloader_mnist,
        attack_type='apgd',
        eps=0.3,
        output_path='adversarial_apgd_dlr.npz',
        device=device,
        n_iter=100,
        n_restarts=5,
        norm='Linf',
        loss='dlr'
    )

    # Example 2: Multi-restart PGD Attack
    # print("\n" + "="*60)
    # print("Example 2: Multi-restart PGD Attack")
    # print("="*60)

    # pgd_data = generate_adversarial_dataset(
    #     pipeline=full_pipeline,
    #     dataloader=dataloader,
    #     attack_type='pgd',
    #     eps=0.3,
    #     output_path='adversarial_pgd_mr.npz',
    #     device=device,
    #     alpha=0.01,
    #     n_iter=100,
    #     n_restarts=10,
    #     norm='Linf'
    # )

    # Example 3: AutoAttack (Standard Version)
    print("\n" + "="*60)
    print("Example 3: AutoAttack (Standard)")
    print("="*60)

    autoattack_data = generate_adversarial_dataset(
        pipeline=full_pipeline,
        dataloader=dataloader_mnist,
        attack_type='autoattack',
        eps=0.3,
        output_path='adversarial_autoattack_standard.npz',
        device=device,
        norm='Linf',
        version='standard'
    )

    # Example 4: AutoAttack with higher restarts
    # print("\n" + "="*60)
    # print("Example 4: AutoAttack (Plus Version - More Aggressive)")
    # print("="*60)

    # autoattack_plus_data = generate_adversarial_dataset(
    #     pipeline=full_pipeline,
    #     dataloader=dataloader,
    #     attack_type='autoattack',
    #     eps=0.3,
    #     output_path='adversarial_autoattack_plus.npz',
    #     device=device,
    #     norm='Linf',
    #     version='plus'  # More restarts and attacks
    # )