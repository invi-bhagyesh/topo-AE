import random
import sys
import time
import string
import shutil
from nltk.metrics import edit_distance
import numpy as np
import torch
import os
import cv2
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchmetrics import StructuralSimilarityIndexMeasure

from dataset import test_adv_dataset
from STR_modules.feature_extraction import VGG_FeatureExtractor, ResNet_FeatureExtractor,BasicBlock,ResNet
from STR_modules.transformation import TPS_SpatialTransformerNetwork
from STR_modules.sequence_modeling import BidirectionalLSTM
from STR_modules.prediction import CTCLabelConverter,AttnLabelConverter
from utils import Logger
from STR_modules.model import Model




def tensor_to_numpy(img_tensor):
    """
    Converts a PyTorch tensor of shape [C, H, W] or [B, C, H, W] to a NumPy array of shape [H, W, C].
    Use img_tensor[0] if the input is a batch.
    """
    img = img_tensor.cpu().clamp(0, 1).detach().numpy()     # move to CPU, detach, clamp to [0,1], to numpy
    img = np.transpose(img, (1, 2, 0))                     # rearrange to [H, W, C]
    return img

# Helper to create directories if not exist
def makedirs(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            print(f'cannot create dirs: {path}')
            sys.exit(0)

# Parse line from results file
def process_line(line):
    adv_img_path, recog_result = line.split(':')
    label, adv_preds = recog_result.split('--->')
    adv_preds = adv_preds.strip('\n')
    return adv_preds, label, adv_img_path

# Main testing function adapted from your code
def test(opt):

    # Model and converter setup
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    model = Model(opt).to(opt.device)
    print(f'Loading STR model from "{opt.str_model}" as the target model!')
    model.load_state_dict(torch.load(opt.str_model, map_location=opt.device), strict=False)
    model.eval()

    # Prepare output directories
    makedirs(opt.output)
    str_name = opt.str_model.split('/')[-1].split('-')[0]
    test_output_path = os.path.join(opt.output, opt.attack_name, str_name)
    attack_success_result = os.path.join(test_output_path, 'attack_success_result.txt')
    save_success_adv = os.path.join(test_output_path, 'attack-success-adv')
    makedirs(test_output_path)
    makedirs(save_success_adv)

    log_file = os.path.join(test_output_path, 'test.log')
    sys.stdout = Logger(log_file)

    # Dataset and dataloader
    dataset = test_adv_dataset(opt.imgH, opt.imgW, "/kaggle/input/invi_str_model/pytorch/default/8/data/data/protego/wmadv")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=1
    )

    result = dict()
    for i, data in enumerate(dataloader):
        adv_img = data[0] if opt.b else data[1]
        adv_img = adv_img.to(opt.device)
        

        reconstructed_img = reformer(adv_img)
        
        label = data[2]
        adv_index = data[3][0]
        adv_path = data[5][0]

        length_for_pred = torch.IntTensor([opt.batch_max_length] * opt.batch_size).to(opt.device)
        text_for_pred = torch.LongTensor(opt.batch_size, opt.batch_max_length + 1).fill_(0).to(opt.device)

        if 'CTC' in opt.Prediction:
            preds = model(reconstructed_img, text_for_pred).log_softmax(2)
            preds_size = torch.IntTensor([preds.size(1)] * opt.batch_size)
            _, preds_index = preds.permute(1, 0, 2).max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            preds_output = converter.decode(preds_index.data, preds_size)
            preds_output = preds_output[0]
            result[adv_index] = f'{adv_path}:{label[0]}--->{preds_output}\n'
        else:  # Attention
            preds = model(reconstructed_img, text_for_pred, is_train=False)
            _, preds_index = preds.max(2)
            preds_output = converter.decode(preds_index, length_for_pred)
            preds_output = preds_output[0]
            preds_output = preds_output[:preds_output.find('[s]')]
            result[adv_index] = f'{adv_path}:{label[0]}--->{preds_output}\n'
        if i < 10:
            orig_np = tensor_to_numpy(adv_img[0])
            recon_np = tensor_to_numpy(reconstructed_img[0])
            
            plt.figure(figsize=(8,4))
            plt.subplot(1,2,1)
            plt.imshow(orig_np)
            plt.title(f'Original Adv Image\nLabel: {label[0]}')
            plt.axis('off')
            
            plt.subplot(1,2,2)
            plt.imshow(recon_np)
            plt.title(f'Reconstruction\nPred: {preds_output}')
            plt.axis('off')
            
            plt.show()
    result = sorted(result.items(), key=lambda x: x[0])
    with open(attack_success_result, 'w+') as f:
        for item in result:
            f.write(item[1])

    # Calculate Attack Success Rate (ASR) and average edit distance
    with open(attack_success_result, 'r') as f:
        alladv = f.readlines()
    
    total_chars = 0
    char_errors = 0
    ED_sum = 0
    attack_success_num = 0  # still counts number of words with any error
    
    for line in alladv:
        adv_preds, label, adv_img_path = process_line(line)
    
        # Count characters for stats
        total_chars += len(label)
        
        # Count char errors as sum of mismatched characters
        # Compare strings at character level up to shorter string length
        min_len = min(len(label), len(adv_preds))
        char_diff = sum(1 for i in range(min_len) if label[i] != adv_preds[i])
        char_diff += abs(len(label) - len(adv_preds))  # add length difference as errors
        char_errors += char_diff
    
        # Word-level error for reference (number of words differing)
        if adv_preds != label:
            attack_success_num += 1
            shutil.copy(adv_img_path, save_success_adv)
            ED_sum += edit_distance(label, adv_preds)

    char_level_asr = char_errors / total_chars if total_chars > 0 else 0
    
    print(f'Character-level Attack Success Rate: {char_level_asr * 100:.2f} %')
    print(f'Word-level Attack Success Rate: {attack_success_num / len(dataset) * 100:.2f} %')
    
    if attack_success_num != 0:
        ED_num_avr = ED_sum / attack_success_num
        print(f'Average Edit_distance per word: {ED_num_avr:.2f}')


class Opt:
    def __init__(self):
        self.output = 'res-BlackModelTest/up5a'
        self.attack_name = 'baseline_attack'  # set this accordingly
        self.adv_img = '/kaggle/input/invi_str_model/pytorch/default/8/data/data/protego/test'  # set your adversarial image path here
        self.b = False
        self.batch_size = 1
        self.img_channel = 3
        self.batch_max_length = 25
        self.imgH = 32
        self.imgW = 100
        self.character = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.sensitive = True
        self.num_class = 63
        self.str_model = '/kaggle/input/invi_str_model/pytorch/default/8/CRNN_VGG_BiLSTM_CTC_model.pth'
        self.Transformation = 'None'
        self.FeatureExtraction = 'VGG'
        self.SequenceModeling = 'BiLSTM'
        self.Prediction = 'CTC'
        self.num_fiducial = 20
        self.input_channel = 3
        self.output_channel = 512
        self.hidden_size = 256
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

opt = Opt()
print(opt.__dict__)


class LatentReformer(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, in_channels, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
    


def train_reformer(opt, model, converter, lambda_ssim=0.1, alpha=1.0, beta=1.0, num_epochs=10):
    device = opt.device
    reformer = LatentReformer(in_channels=3).to(device)
    optimizer = optim.Adam(reformer.parameters(), lr=1e-3)
    mse_criterion = nn.MSELoss()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ctc_criterion = nn.CTCLoss(zero_infinity=True).to(device)

    # Freeze classifier weights so only reformer updates
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    

    dataset = test_adv_dataset(opt.imgH, opt.imgW, opt.adv_img)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=1)

    output_dir = './reformer_output_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for epoch in range(num_epochs):
        reformer.train()
        running_loss = 0.0
        orig_img = None
        recon_img = None

        for i, data in enumerate(dataloader):
            adv_img = data[0] if opt.b else data[1]
            adv_img = adv_img.to(device)
            label = data[2]
            optimizer.zero_grad()

            # Forward through reformer
            output_img = reformer(adv_img)

            # Reconstruction loss
            mse_loss = mse_criterion(output_img, adv_img)
            ssim_loss = 1 - ssim_metric(output_img, adv_img)
            recon_loss = mse_loss + lambda_ssim * ssim_loss

            
            # Prepare classifier inputs
            length_for_pred = torch.IntTensor([opt.batch_max_length] * opt.batch_size).to(device)
            text_for_pred = torch.LongTensor(opt.batch_size, opt.batch_max_length + 1).fill_(0).to(device)
            
            with torch.no_grad():
                preds = model(output_img, text_for_pred)
            
            ctc_loss = torch.tensor(0.0, device=device)
            # Prepare ground truth for CTC
            if 'CTC' in opt.Prediction:
                text_for_loss, length_for_loss = converter.encode(label, batch_max_length=opt.batch_max_length)
                preds_size = torch.IntTensor([preds.size(1)] * opt.batch_size).to(device)
                ctc_loss = ctc_criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

            # Backpropagate only reconstruction losses (no grad from classifier)
            loss = alpha * recon_loss  # optionally monitor ctc_loss but DON'T backpropagate it
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i == 0:
                orig_img = adv_img.cpu()
                recon_img = output_img.detach().cpu()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Reconstruction Loss: {avg_loss:.4f} CTC Loss (no grad): {ctc_loss.item():.4f}")

        # Visualization
        orig_img_np = orig_img[0].clamp(0, 1).permute(1, 2, 0).numpy()
        recon_img_np = recon_img[0].clamp(0, 1).permute(1, 2, 0).numpy()

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title('Original Adversarial')
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
    return reformer

opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(opt).to(opt.device)
converter = CTCLabelConverter(opt.character) or AttnLabelConverter(opt.character)

reformer = train_reformer(opt,model,converter)

time_st = time.time()
test(opt)
time_end = time.time()
print(f'Testing time: {time_end - time_st:.2f} seconds')

