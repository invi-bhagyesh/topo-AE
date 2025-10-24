import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchmetrics import StructuralSimilarityIndexMeasure
import numpy as np
import random
from .utils import set_seed



def train_reformer(opt, model, converter, lambda_ssim=0.1, alpha=1.0, beta=1.0, num_epochs=5):
    device = opt.device
    
    reformer = LatentReformer(in_channels=3, bottleneck_H=32, bottleneck_W=100).to(device)
    latent_net = LatentNet(latent_dim=16, max_length=15, bottleneck_H=32, bottleneck_W=100).to(device)

    optimizer = optim.Adam(
        list(reformer.parameters()) + list(latent_net.parameters()),
        lr=1e-3
    )

    mse_criterion = nn.MSELoss()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ctc_criterion = nn.CTCLoss(zero_infinity=True).to(device)

    # Freeze classifier weights; only train reformer & latent_net
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    data = np.load(opt.adv_img, allow_pickle=True)
    keys = data.files

    output_dir = './reformer_output_images'
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(num_epochs):
        reformer.train()
        latent_net.train()
        running_loss = 0.0
        sample_to_show = random.randint(0, len(keys) - 1)
        orig_img = None
        recon_img = None

        for i, key in enumerate(keys):
            sample = data[key].item()
            adv_img_np = sample['image']  # H x W x C numpy
            adv_img = torch.from_numpy(adv_img_np).float() / 255.0
            adv_img = adv_img.permute(2, 0, 1).unsqueeze(0).to(device)  # B=1, C, H, W

            latents_np = sample['latents']  # N x 16
            latents_tensor = torch.from_numpy(latents_np).unsqueeze(0).to(device)  # B=1, N, 16

            label = [sample['word_label']]  # wrap label in list for converter

            optimizer.zero_grad()

            latents_expanded = latent_net(latents_tensor)
            output_img = reformer(adv_img, latents_expanded)

            mse_loss = mse_criterion(output_img, adv_img)
            ssim_loss = 1 - ssim_metric(output_img, adv_img)
            recon_loss = mse_loss + lambda_ssim * ssim_loss

            # Prepare classifier inputs for CTC loss
            length_for_pred = torch.IntTensor([opt.batch_max_length]).to(device)
            text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)

            with torch.no_grad():
                preds = model(output_img, text_for_pred)

            ctc_loss_val = torch.tensor(0.0, device=device)
            if 'CTC' in opt.Prediction:
                text_for_loss, length_for_loss = converter.encode(label, batch_max_length=opt.batch_max_length)
                preds_size = torch.IntTensor([preds.size(1)]).to(device)
                ctc_loss_val = ctc_criterion(preds.log_softmax(2).permute(1, 0, 2),
                                            text_for_loss,
                                            preds_size,
                                            length_for_loss)

            loss = alpha * recon_loss + beta * ctc_loss_val
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i == sample_to_show:
                orig_img = adv_img.cpu()
                recon_img = output_img.detach().cpu()

        avg_loss = running_loss / len(keys)
        print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {avg_loss:.4f}, CTC Loss: {ctc_loss_val.item():.4f}")

        orig_img_np = orig_img[0].clamp(0, 1).permute(1, 2, 0).numpy()
        recon_img_np = recon_img[0].clamp(0, 1).permute(1, 2, 0).numpy()

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title('Original Adv Image')
        plt.imshow(orig_img_np)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Reconstruction')
        plt.imshow(recon_img_np)
        plt.axis('off')

        plt.suptitle(f'Epoch {epoch+1}')
        plt.savefig(os.path.join(output_dir, f'reconstruction_epoch_{epoch+1}.png'))
        plt.show()

        ssim_val = ssim_metric(recon_img.to(device), orig_img.to(device)).item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Sample SSIM: {ssim_val:.4f}")

    print("Training completed.")
    return reformer, latent_net



set_seed(42)

reformer,latent_net = train_reformer(opt,model,converter)