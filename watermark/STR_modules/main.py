


"""
USAGE EXAMPLES:

# Pipeline 1: Test wmadv images directly on STR
python script.py --pipeline 1 --mode test \
    --str_model ./models/RARE-TPS-VGG-BiLSTM-Attn-sensitive.pth \
    --wmadv_path ./data/wmadv \
    --npz_adv_path ./data/adv.npz

# Pipeline 2: Train Reformer and test
python script.py --pipeline 2 --mode both \
    --str_model ./models/RARE-TPS-VGG-BiLSTM-Attn-sensitive.pth \
    --npz_train_path ./data/train.npz \
    --npz_adv_path ./data/adv.npz \
    --num_epochs 10

# Pipeline 2: Only test (requires pre-trained reformer)
python script.py --pipeline 2 --mode test \
    --str_model ./models/RARE-TPS-VGG-BiLSTM-Attn-sensitive.pth \
    --npz_adv_path ./data/adv.npz

# Pipeline 3: Train LatentNet + Reformer and test
python script.py --pipeline 3 --mode both \
    --str_model ./models/RARE-TPS-VGG-BiLSTM-Attn-sensitive.pth \
    --npz_train_path ./data/train.npz \
    --npz_adv_path ./data/adv.npz \
    --num_epochs 10 \
    --alpha 1.0 --beta 1.0

# Pipeline 3: Only test
python script.py --pipeline 3 --mode test \
    --str_model ./models/RARE-TPS-VGG-BiLSTM-Attn-sensitive.pth \
    --npz_adv_path ./data/adv.npz

# Pipeline 4: Warmup + Joint training and test
python script.py --pipeline 4 --mode both \
    --str_model ./models/RARE-TPS-VGG-BiLSTM-Attn-sensitive.pth \
    --npz_train_path ./data/train.npz \
    --npz_adv_path ./data/adv.npz \
    --num_epochs 10 \
    --warmup_epochs 2 \
    --alpha 1.0 --beta 0.5

# Pipeline 4: Only test
python script.py --pipeline 4 --mode test \
    --str_model ./models/RARE-TPS-VGG-BiLSTM-Attn-sensitive.pth \
    --npz_adv_path ./data/adv.npz

# Advanced options
python script.py --pipeline 3 --mode both \
    --str_model ./models/model.pth \
    --npz_train_path ./data/train.npz \
    --npz_adv_path ./data/adv.npz \
    --num_epochs 15 \
    --batch_size 8 \
    --lr 5e-4 \
    --lambda_ssim 0.2 \
    --alpha 1.5 \
    --beta 0.8 \
    --imgH 32 --imgW 100 \
    --output ./my_results \
    --visualize_samples 10

PIPELINE DESCRIPTIONS:

Pipeline 1: wmadv → STR
- Tests adversarial images directly on the STR model
- No defense mechanism
- Baseline for comparison
- Requires: --wmadv_path

Pipeline 2: npz → Reformer → STR
- Trains a basic image reformer (encoder-decoder)
- Reformer tries to clean adversarial images
- Does NOT use latent information
- Requires: --npz_train_path (for training), --npz_adv_path (for testing)
- Saves: ./pipeline2_reformer.pth

Pipeline 3: npz → LatentNet + Reformer → STR
- Trains LatentNet to expand latent sequences to spatial features
- Trains latent-conditioned Reformer that uses expanded latents
- Joint training from scratch
- Requires: --npz_train_path (for training), --npz_adv_path (for testing)
- Saves: ./pipeline3_reformer.pth, ./pipeline3_latentnet.pth

Pipeline 4: Warmup + Joint Training
- Phase 1: Warmup train basic Reformer (no latents)
- Phase 2: Convert to latent-conditioned, initialize from warmup weights
- Phase 3: Joint train LatentNet + Reformer
- Better convergence than Pipeline 3
- Requires: --npz_train_path (for training), --npz_adv_path (for testing)
- Saves: ./pipeline4_reformer.pth, ./pipeline4_latentnet.pth

NOTES:
- STR model is always pretrained and frozen
- Training data (npz_train_path) should contain clean or weakly adversarial images
- Test data (npz_adv_path) contains strong adversarial examples
- All models save in current directory by default
- Results save in --output directory (default: ./results)
- Use --mode train to only train, --mode test to only test, --mode both for both
"""
import random
import sys 
import os
import time
import shutil
import argparse
from nltk.metrics import edit_distance
import numpy as np
import torch
import cv2
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchmetrics import StructuralSimilarityIndexMeasure
import sys
import os
current_dir = os.getcwd()  # current notebook or script folder
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from feature_extraction import VGG_FeatureExtractor, ResNet_FeatureExtractor, BasicBlock, ResNet
from transformation import TPS_SpatialTransformerNetwork
from sequence_modeling import BidirectionalLSTM
from prediction import CTCLabelConverter, AttnLabelConverter
from utils import Logger
from model import Model


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def tensor_to_numpy(img_tensor):
    """Convert PyTorch tensor [C, H, W] to numpy [H, W, C]"""
    img = img_tensor.cpu().clamp(0, 1).detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    return img

def makedirs(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            print(f'Cannot create dirs: {path}')
            sys.exit(0)

def process_line(line):
    """Parse result line: 'path:label--->prediction'"""
    adv_img_path, recog_result = line.split(':')
    label, adv_preds = recog_result.split('--->')
    adv_preds = adv_preds.strip('\n')
    return adv_preds, label, adv_img_path


# ============================================================================
# DATASET CLASSES
# ============================================================================

class WmAdvDataset(Dataset):
    """Dataset for wmadv adversarial images (Pipeline 1)"""
    def __init__(self, height, width, img_path):      
        self.height = height
        self.width = width
        self.img_path = img_path
        self.dataset = []
        
        img = [] 
        for i, j, k in os.walk(self.img_path):            
            for file in k:
                file_name = os.path.join(i, file)
                img.append(file_name)
        self.total_img_name = img
        
        for img_name in self.total_img_name:
            base_name = os.path.basename(img_name)
            parts = base_name.rsplit('_', 2)
            if len(parts) == 3:
                img_index, label, img_adv = parts
                img_adv = img_adv.split('.')
                index_or_advlogo = img_adv[0]
                self.dataset.append([img_name, label, img_index, index_or_advlogo])
            else:
                print(f"Skipping unexpected filename format: {base_name}")
        self.dataset = sorted(self.dataset)

    def __getitem__(self, index):
        img_name, label, img_index, index_or_advlogo = self.dataset[index]        
        IMG = cv2.imread(img_name) 
        IMG = cv2.resize(IMG, (self.width, self.height))
        
        # Binarization processing
        gray = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_b = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        img_b = torch.FloatTensor(img_b) / 255
        img_b = img_b.permute(2, 0, 1)

        img = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
        img = torch.FloatTensor(img) / 255
        img = img.permute(2, 0, 1)

        return img_b, img, label, img_index, index_or_advlogo, img_name

    def __len__(self):
        return len(self.dataset)


class NPZDataset(Dataset):
    """Dataset for NPZ files (Pipelines 2, 3, 4)"""
    def __init__(self, npz_path):
        self.data = np.load(npz_path, allow_pickle=True)
        self.keys = self.data.files

    def __getitem__(self, index):
        key = self.keys[index]
        sample = self.data[key].item()
        
        # Image
        image_np = sample['image']  # H x W x C
        image_tensor = torch.from_numpy(image_np).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # C x H x W
        
        # Latents
        latents_np = sample['latents']  # N x 16
        latents_tensor = torch.from_numpy(latents_np)
        
        # Label
        label = sample['word_label']
        
        return image_tensor, latents_tensor, label, key

    def __len__(self):
        return len(self.keys)


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class LatentReformer(nn.Module):
    """Basic Reformer without latent conditioning (Pipeline 2)"""
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

    def forward(self, x, latents=None):
        z = self.encoder(x)
        # Latents ignored in basic version
        out = self.decoder(z)
        return out


class LatentConditionedReformer(nn.Module):
    """Reformer with latent conditioning (Pipelines 3, 4)"""
    def __init__(self, in_channels=3, latent_channels=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU()
        )
        
        # Combine bottleneck with latents
        self.fusion = nn.Sequential(
            nn.Conv2d(256 + latent_channels, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, in_channels, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x, latents):
        z = self.encoder(x)
        # Concatenate latents with bottleneck
        z_combined = torch.cat([z, latents], dim=1)
        z_fused = self.fusion(z_combined)
        out = self.decoder(z_fused)
        return out


class LatentNet(nn.Module):
    """Expand latent sequences to spatial feature maps (Pipelines 3, 4)"""
    def __init__(self, latent_dim=16, max_length=15, bottleneck_H=32, bottleneck_W=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_length = max_length
        self.bottleneck_H = bottleneck_H
        self.bottleneck_W = bottleneck_W
        self.expand_dim = bottleneck_H * bottleneck_W

        self.fc = nn.Sequential(
            nn.Linear(latent_dim * max_length, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, latent_dim * self.expand_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, latents):
        B, N, D = latents.shape
        assert D == self.latent_dim, f"Expected latent_dim={self.latent_dim}, got {D}"
        
        # Pad if necessary
        if N < self.max_length:
            pad_size = self.max_length - N
            padding = torch.zeros(B, pad_size, D, device=latents.device, dtype=latents.dtype)
            latents = torch.cat([latents, padding], dim=1)
        
        x = latents.reshape(B, -1)
        x = self.fc(x)
        x = x.view(B, self.latent_dim, self.bottleneck_H, self.bottleneck_W)
        return x


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_predictions(result_file, save_success_adv):
    """Compute metrics from result file"""
    with open(result_file, 'r') as f:
        alladv = f.readlines()

    total_gt_chars = 0
    total_pred_chars = 0
    correct_chars = 0
    false_positive = 0
    false_negative = 0
    ED_sum = 0
    attack_success_num = 0
    
    for line in alladv:
        adv_preds, label, adv_img_path = process_line(line)
        total_gt_chars += len(label)
        total_pred_chars += len(adv_preds)
        
        min_len = min(len(label), len(adv_preds))
        for j in range(min_len):
            if adv_preds[j] == label[j]:
                correct_chars += 1
            else:
                false_positive += 1
                false_negative += 1
        
        if len(adv_preds) > len(label):
            false_positive += (len(adv_preds) - len(label))
        if len(label) > len(adv_preds):
            false_negative += (len(label) - len(adv_preds))
        
        if adv_preds != label:
            attack_success_num += 1
            try:
                shutil.copy(adv_img_path, save_success_adv)
            except Exception:
                pass
            ED_sum += edit_distance(label, adv_preds)
    
    char_level_accuracy = correct_chars / total_gt_chars if total_gt_chars > 0 else 0
    char_level_precision = correct_chars / total_pred_chars if total_pred_chars > 0 else 0
    char_level_recall = correct_chars / total_gt_chars if total_gt_chars > 0 else 0
    
    print(f'\n{"="*60}')
    print(f'EVALUATION METRICS')
    print(f'{"="*60}')
    print(f'Character-level Accuracy:  {char_level_accuracy * 100:.2f}%')
    print(f'Character-level Precision: {char_level_precision * 100:.2f}%')
    print(f'Character-level Recall:    {char_level_recall * 100:.2f}%')
    print(f'Word-level Attack Success: {attack_success_num / len(alladv) * 100:.2f}%')
    
    if attack_success_num != 0:
        ED_num_avr = ED_sum / attack_success_num
        print(f'Average Edit Distance:     {ED_num_avr:.2f}')
    print(f'{"="*60}\n')


# ============================================================================
# PIPELINE 1: wmadv → STR
# ============================================================================

def pipeline1_test_wmadv(opt):
    """Test adversarial images directly on STR model"""
    print("\n" + "="*60)
    print("PIPELINE 1: wmadv → STR")
    print("="*60 + "\n")
    
    # Setup converter and model
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    
    model = Model(opt).to(opt.device)
    print(f'Loading STR model from: {opt.str_model}')
    model.load_state_dict(torch.load(opt.str_model, map_location=opt.device), strict=False)
    model.eval()
    
    # Prepare output directories
    makedirs(opt.output)
    str_name = opt.str_model.split('/')[-1].split('-')[0]
    test_output_path = os.path.join(opt.output, 'pipeline1_wmadv', str_name)
    attack_success_result = os.path.join(test_output_path, 'attack_success_result.txt')
    save_success_adv = os.path.join(test_output_path, 'attack-success-adv')
    makedirs(test_output_path)
    makedirs(save_success_adv)
    
    # Dataset and dataloader
    dataset = WmAdvDataset(opt.imgH, opt.imgW, opt.wmadv_path)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=1)
    
    result = dict()
    print(f"Testing {len(dataset)} adversarial images...\n")
    
    for i, data in enumerate(dataloader):
        adv_img = data[0] if opt.use_binary else data[1]
        adv_img = adv_img.to(opt.device)
        label = data[2]
        adv_index = data[3][0]
        adv_path = data[5][0]
        
        length_for_pred = torch.IntTensor([opt.batch_max_length] * opt.batch_size).to(opt.device)
        text_for_pred = torch.LongTensor(opt.batch_size, opt.batch_max_length + 1).fill_(0).to(opt.device)
        
        with torch.no_grad():
            if 'CTC' in opt.Prediction:
                preds = model(adv_img, text_for_pred).log_softmax(2)
                preds_size = torch.IntTensor([preds.size(1)] * opt.batch_size)
                _, preds_index = preds.permute(1, 0, 2).max(2)
                preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
                preds_output = converter.decode(preds_index.data, preds_size)[0]
            else:
                preds = model(adv_img, text_for_pred, is_train=False)
                _, preds_index = preds.max(2)
                preds_output = converter.decode(preds_index, length_for_pred)[0]
                preds_output = preds_output[:preds_output.find('[s]')]
        
        result[adv_index] = f'{adv_path}:{label[0]}--->{preds_output}\n'
        
        if i < 5:
            print(f"Sample {i+1}: Label='{label[0]}', Pred='{preds_output}'")
    
    # Write results
    result = sorted(result.items(), key=lambda x: x[0])
    with open(attack_success_result, 'w+') as f:
        for item in result:
            f.write(item[1])
    
    # Evaluate
    evaluate_predictions(attack_success_result, save_success_adv)


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='OCR Defense Pipeline System')
    
    # Pipeline selection
    parser.add_argument('--pipeline', type=int, required=True, choices=[1, 2, 3, 4],
                        help='Pipeline mode: 1=wmadv→STR, 2=npz→Reformer→STR, '
                             '3=npz→LatentNet+Reformer→STR, 4=warmup+joint training')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test', 'both'],
                        help='Mode: train, test, or both (default: test)')
    
    # Paths
    parser.add_argument('--output', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--str_model', type=str, required=True,
                        help='Path to pretrained STR model')
    parser.add_argument('--wmadv_path', type=str, default='',
                        help='Path to wmadv adversarial images (Pipeline 1)')
    parser.add_argument('--npz_train_path', type=str, default='',
                        help='Path to NPZ training data (Pipelines 2, 3, 4)')
    parser.add_argument('--npz_adv_path', type=str, required=True,
                        help='Path to NPZ adversarial test data (Pipelines 2, 3, 4)')
    
    # Model architecture parameters
    parser.add_argument('--Transformation', type=str, default='TPS',
                        help='Transformation stage: None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='VGG',
                        help='Feature extraction: VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM',
                        help='Sequence modeling: None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='Attn',
                        help='Prediction: CTC|Attn')
    
    # STR model parameters
    parser.add_argument('--num_fiducial', type=int, default=20,
                        help='Number of fiducial points for TPS')
    parser.add_argument('--input_channel', type=int, default=3,
                        help='Number of input channels')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='Number of output channels')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Hidden size for sequence modeling')
    
    # Data parameters
    parser.add_argument('--imgH', type=int, default=32,
                        help='Height of input image')
    parser.add_argument('--imgW', type=int, default=100,
                        help='Width of input image')
    parser.add_argument('--img_channel', type=int, default=3,
                        help='Number of image channels')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',
                        help='Character set')
    parser.add_argument('--sensitive', action='store_true',
                        help='Case sensitive')
    parser.add_argument('--batch_max_length', type=int, default=25,
                        help='Maximum text length')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                        help='Number of warmup epochs for Pipeline 4')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    
    # Loss weights
    parser.add_argument('--lambda_ssim', type=float, default=0.1,
                        help='Weight for SSIM loss')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for reconstruction loss')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Weight for CTC loss')
    
    # Other options
    parser.add_argument('--use_binary', action='store_true',
                        help='Use binarized images (Pipeline 1)')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of data loading workers')
    parser.add_argument('--visualize_samples', type=int, default=5,
                        help='Number of samples to visualize during testing')
    
    args = parser.parse_args()
    
    # Set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Validate paths based on pipeline
    if args.pipeline == 1 and args.mode in ['test', 'both']:
        if not args.wmadv_path:
            parser.error("Pipeline 1 requires --wmadv_path")
    
    if args.pipeline in [2, 3, 4]:
        if args.mode in ['train', 'both'] and not args.npz_train_path:
            parser.error(f"Pipeline {args.pipeline} training requires --npz_train_path")
    
    return args


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    args = get_args()
    
    print("\n" + "="*60)
    print(f"STARTING PIPELINE {args.pipeline}")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Device: {args.device}")
    print(f"STR Model: {args.str_model}")
    print("="*60 + "\n")
    
    # Setup converter
    if 'CTC' in args.Prediction:
        converter = CTCLabelConverter(args.character)
    else:
        converter = AttnLabelConverter(args.character)
    args.num_class = len(converter.character)
    
    # Load STR model
    model = Model(args).to(args.device)
    model.load_state_dict(torch.load(args.str_model, map_location=args.device), strict=False)
    model.eval()
    
    # Execute pipeline
    if args.pipeline == 1:
        # Pipeline 1: wmadv → STR
        if args.mode in ['test', 'both']:
            pipeline1_test_wmadv(args)
    
    elif args.pipeline == 2:
        # Pipeline 2: npz → Reformer → STR
        if args.mode in ['train', 'both']:
            pipeline2_train_reformer(args, model, converter, num_epochs=args.num_epochs)
        if args.mode in ['test', 'both']:
            pipeline2_test_npz_reformer(args)
    
    elif args.pipeline == 3:
        # Pipeline 3: npz → LatentNet + Reformer → STR
        if args.mode in ['train', 'both']:
            pipeline3_train_latent_reformer(args, model, converter, num_epochs=args.num_epochs)
        if args.mode in ['test', 'both']:
            pipeline3_test_npz_latent(args)
    
    elif args.pipeline == 4:
        # Pipeline 4: warmup + joint training
        if args.mode in ['train', 'both']:
            pipeline4_warmup_and_train(args, model, converter)
        if args.mode in ['test', 'both']:
            pipeline4_test_npz_latent(args)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()


# ============================================================================
# PIPELINE 2: npz → Reformer → STR
# ============================================================================

def pipeline2_train_reformer(opt, model, converter, num_epochs=10):
    """Train basic Reformer on NPZ training data"""
    print("\n" + "="*60)
    print("PIPELINE 2 - TRAINING: npz → Reformer")
    print("="*60 + "\n")
    
    device = opt.device
    reformer = LatentReformer(in_channels=3).to(device)
    optimizer = optim.Adam(reformer.parameters(), lr=1e-3)
    mse_criterion = nn.MSELoss()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # Freeze STR model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # Load training data
    dataset = NPZDataset(opt.npz_train_path)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=1)
    
    output_dir = './pipeline2_reformer_output'
    makedirs(output_dir)
    
    print(f"Training on {len(dataset)} samples for {num_epochs} epochs\n")
    
    for epoch in range(num_epochs):
        reformer.train()
        running_loss = 0.0
        sample_idx = random.randint(0, len(dataset) - 1)
        
        for i, (image, latents, label, key) in enumerate(dataloader):
            image = image.to(device)
            optimizer.zero_grad()
            
            # Forward through reformer (latents not used in basic version)
            output_img = reformer(image)
            
            # Reconstruction loss
            mse_loss = mse_criterion(output_img, image)
            ssim_loss = 1 - ssim_metric(output_img, image)
            loss = mse_loss + opt.lambda_ssim * ssim_loss
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Save sample for visualization
            if i == sample_idx:
                sample_orig = image[0].cpu()
                sample_recon = output_img[0].detach().cpu()
                sample_label = label[0]
        
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        # Visualize
        if epoch % max(1, num_epochs // 5) == 0:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(tensor_to_numpy(sample_orig))
            axes[0].set_title(f'Original\nLabel: {sample_label}')
            axes[0].axis('off')
            axes[1].imshow(tensor_to_numpy(sample_recon))
            axes[1].set_title('Reconstruction')
            axes[1].axis('off')
            plt.suptitle(f'Epoch {epoch+1}')
            plt.savefig(os.path.join(output_dir, f'epoch_{epoch+1}.png'))
            plt.close()
    
    # Save model
    torch.save(reformer.state_dict(), './pipeline2_reformer.pth')
    print("\nTraining completed. Model saved to './pipeline2_reformer.pth'\n")
    return reformer


def pipeline2_test_npz_reformer(opt):
    """Test NPZ adversarial data through Reformer → STR"""
    print("\n" + "="*60)
    print("PIPELINE 2 - TESTING: npz → Reformer → STR")
    print("="*60 + "\n")
    
    # Setup converter and model
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    
    model = Model(opt).to(opt.device)
    model.load_state_dict(torch.load(opt.str_model, map_location=opt.device), strict=False)
    model.eval()
    
    # Load reformer
    reformer = LatentReformer(in_channels=3).to(opt.device)
    reformer.load_state_dict(torch.load('./pipeline2_reformer.pth', map_location=opt.device))
    reformer.eval()
    
    # Prepare output
    str_name = opt.str_model.split('/')[-1].split('-')[0]
    test_output_path = os.path.join(opt.output, 'pipeline2_npz_reformer', str_name)
    attack_success_result = os.path.join(test_output_path, 'attack_success_result.txt')
    save_success_adv = os.path.join(test_output_path, 'attack-success-adv')
    makedirs(test_output_path)
    makedirs(save_success_adv)
    
    # Load adversarial data
    dataset = NPZDataset(opt.npz_adv_path)
    
    result = dict()
    print(f"Testing {len(dataset)} adversarial samples...\n")
    
    for i in range(len(dataset)):
        image, latents, label, key = dataset[i]
        image = image.unsqueeze(0).to(opt.device)
        
        # Reconstruct
        with torch.no_grad():
            reconstructed_img = reformer(image)
        
        # Predict
        length_for_pred = torch.IntTensor([opt.batch_max_length]).to(opt.device)
        text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(opt.device)
        
        with torch.no_grad():
            if 'CTC' in opt.Prediction:
                preds = model(reconstructed_img, text_for_pred).log_softmax(2)
                preds_size = torch.IntTensor([preds.size(1)])
                _, preds_index = preds.permute(1, 0, 2).max(2)
                preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
                preds_output = converter.decode(preds_index.data, preds_size)[0]
            else:
                preds = model(reconstructed_img, text_for_pred, is_train=False)
                _, preds_index = preds.max(2)
                preds_output = converter.decode(preds_index, length_for_pred)[0]
                preds_output = preds_output[:preds_output.find('[s]')]
        
        result[i] = f'{key}:{label}--->{preds_output}\n'
        
        if i < 5:
            print(f"Sample {i+1}: Label='{label}', Pred='{preds_output}'")
    
    # Write results
    result = sorted(result.items(), key=lambda x: x[0])
    with open(attack_success_result, 'w+') as f:
        for _, line in result:
            f.write(line)
    
    # Evaluate
    evaluate_predictions(attack_success_result, save_success_adv)


# ============================================================================
# PIPELINE 3: npz → LatentNet + Reformer → STR
# ============================================================================

def pipeline3_train_latent_reformer(opt, model, converter, num_epochs=10):
    """Train LatentNet + Latent-Conditioned Reformer jointly"""
    print("\n" + "="*60)
    print("PIPELINE 3 - TRAINING: npz → LatentNet + Reformer")
    print("="*60 + "\n")
    
    device = opt.device
    reformer = LatentConditionedReformer(in_channels=3, latent_channels=16).to(device)
    latent_net = LatentNet(latent_dim=16, max_length=15, 
                           bottleneck_H=opt.imgH, bottleneck_W=opt.imgW).to(device)
    
    optimizer = optim.Adam(
        list(reformer.parameters()) + list(latent_net.parameters()),
        lr=1e-3
    )
    
    mse_criterion = nn.MSELoss()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ctc_criterion = nn.CTCLoss(zero_infinity=True).to(device)
    
    # Freeze STR model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # Load training data
    dataset = NPZDataset(opt.npz_train_path)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=1)
    
    output_dir = './pipeline3_latent_reformer_output'
    makedirs(output_dir)
    
    print(f"Training on {len(dataset)} samples for {num_epochs} epochs\n")
    
    for epoch in range(num_epochs):
        reformer.train()
        latent_net.train()
        running_loss = 0.0
        sample_idx = random.randint(0, len(dataset) - 1)
        
        for i, (image, latents, label, key) in enumerate(dataloader):
            image = image.to(device)
            latents = latents.to(device)
            
            # Ensure latents has batch dimension
            if latents.dim() == 2:
                latents = latents.unsqueeze(0)
            
            optimizer.zero_grad()
            
            # Expand latents and reconstruct
            latents_expanded = latent_net(latents)
            output_img = reformer(image, latents_expanded)
            
            # Reconstruction loss
            mse_loss = mse_criterion(output_img, image)
            ssim_loss = 1 - ssim_metric(output_img, image)
            recon_loss = mse_loss + opt.lambda_ssim * ssim_loss
            
            # CTC loss (optional, for monitoring)
            length_for_pred = torch.IntTensor([opt.batch_max_length]).to(device)
            text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)
            
            with torch.no_grad():
                preds = model(output_img, text_for_pred)
            
            ctc_loss_val = torch.tensor(0.0, device=device)
            if 'CTC' in opt.Prediction:
                text_for_loss, length_for_loss = converter.encode([label[0]], 
                                                                   batch_max_length=opt.batch_max_length)
                preds_size = torch.IntTensor([preds.size(1)]).to(device)
                ctc_loss_val = ctc_criterion(preds.log_softmax(2).permute(1, 0, 2),
                                            text_for_loss, preds_size, length_for_loss)
            
            # Total loss (can add beta * ctc_loss_val if needed)
            loss = opt.alpha * recon_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if i == sample_idx:
                sample_orig = image[0].cpu()
                sample_recon = output_img[0].detach().cpu()
                sample_label = label[0]
        
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, CTC: {ctc_loss_val.item():.4f}")
        
        # Visualize
        if epoch % max(1, num_epochs // 5) == 0:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(tensor_to_numpy(sample_orig))
            axes[0].set_title(f'Original\nLabel: {sample_label}')
            axes[0].axis('off')
            axes[1].imshow(tensor_to_numpy(sample_recon))
            axes[1].set_title('Reconstruction')
            axes[1].axis('off')
            plt.suptitle(f'Epoch {epoch+1}')
            plt.savefig(os.path.join(output_dir, f'epoch_{epoch+1}.png'))
            plt.close()
    
    # Save models
    torch.save(reformer.state_dict(), './pipeline3_reformer.pth')
    torch.save(latent_net.state_dict(), './pipeline3_latentnet.pth')
    print("\nTraining completed. Models saved.\n")
    return reformer, latent_net


def pipeline3_test_npz_latent(opt):
    """Test NPZ adversarial data through LatentNet + Reformer → STR"""
    print("\n" + "="*60)
    print("PIPELINE 3 - TESTING: npz → LatentNet + Reformer → STR")
    print("="*60 + "\n")
    
    # Setup converter and model
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    
    model = Model(opt).to(opt.device)
    model.load_state_dict(torch.load(opt.str_model, map_location=opt.device), strict=False)
    model.eval()
    
    # Load reformer and latent_net
    reformer = LatentConditionedReformer(in_channels=3, latent_channels=16).to(opt.device)
    reformer.load_state_dict(torch.load('./pipeline3_reformer.pth', map_location=opt.device))
    reformer.eval()
    
    latent_net = LatentNet(latent_dim=16, max_length=15, 
                          bottleneck_H=opt.imgH, bottleneck_W=opt.imgW).to(opt.device)
    latent_net.load_state_dict(torch.load('./pipeline3_latentnet.pth', map_location=opt.device))
    latent_net.eval()
    
    # Prepare output
    str_name = opt.str_model.split('/')[-1].split('-')[0]
    test_output_path = os.path.join(opt.output, 'pipeline3_npz_latent', str_name)
    attack_success_result = os.path.join(test_output_path, 'attack_success_result.txt')
    save_success_adv = os.path.join(test_output_path, 'attack-success-adv')
    makedirs(test_output_path)
    makedirs(save_success_adv)
    
    # Load adversarial data
    dataset = NPZDataset(opt.npz_adv_path)
    
    result = dict()
    print(f"Testing {len(dataset)} adversarial samples...\n")
    
    for i in range(len(dataset)):
        image, latents, label, key = dataset[i]
        image = image.unsqueeze(0).to(opt.device)
        latents = latents.to(opt.device)
        
        # Ensure latents has batch dimension
        if latents.dim() == 2:
            latents = latents.unsqueeze(0)
        
        # Reconstruct
        with torch.no_grad():
            latents_expanded = latent_net(latents)
            reconstructed_img = reformer(image, latents_expanded)
        
        # Predict
        length_for_pred = torch.IntTensor([opt.batch_max_length]).to(opt.device)
        text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(opt.device)
        
        with torch.no_grad():
            if 'CTC' in opt.Prediction:
                preds = model(reconstructed_img, text_for_pred).log_softmax(2)
                preds_size = torch.IntTensor([preds.size(1)])
                _, preds_index = preds.permute(1, 0, 2).max(2)
                preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
                preds_output = converter.decode(preds_index.data, preds_size)[0]
            else:
                preds = model(reconstructed_img, text_for_pred, is_train=False)
                _, preds_index = preds.max(2)
                preds_output = converter.decode(preds_index, length_for_pred)[0]
                preds_output = preds_output[:preds_output.find('[s]')]
        
        result[i] = f'{key}:{label}--->{preds_output}\n'
        
        if i < opt.visualize_samples:
            print(f"Sample {i+1}: Label='{label}', Pred='{preds_output}'")
            
            # Visualize
            if i < 3:
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                axes[0].imshow(tensor_to_numpy(image[0]))
                axes[0].set_title(f'Original Adv\nLabel: {label}')
                axes[0].axis('off')
                axes[1].imshow(tensor_to_numpy(reconstructed_img[0]))
                axes[1].set_title(f'Reconstruction\nPred: {preds_output}')
                axes[1].axis('off')
                plt.tight_layout()
                plt.show()
    
    # Write results
    result = sorted(result.items(), key=lambda x: x[0])
    with open(attack_success_result, 'w+') as f:
        for _, line in result:
            f.write(line)
    
    # Evaluate
    evaluate_predictions(attack_success_result, save_success_adv)


# ============================================================================
# PIPELINE 4: Warmup + Joint Training
# ============================================================================

def pipeline4_warmup_and_train(opt, model, converter):
    """Pipeline 4: Warmup train Reformer, then joint fine-tune"""
    print("\n" + "="*60)
    print("PIPELINE 4: Warmup + Joint Training")
    print("="*60 + "\n")
    
    # Phase 1: Warmup - train basic Reformer only
    print("PHASE 1: Warmup training Reformer...")
    print("-" * 60)
    
    device = opt.device
    reformer = LatentReformer(in_channels=3).to(device)
    optimizer_warmup = optim.Adam(reformer.parameters(), lr=opt.lr)
    mse_criterion = nn.MSELoss()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # Freeze STR model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # Load training data
    dataset = NPZDataset(opt.npz_train_path)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, 
                           num_workers=opt.num_workers)
    
    output_dir = './pipeline4_output'
    makedirs(output_dir)
    
    # Warmup training
    for epoch in range(opt.warmup_epochs):
        reformer.train()
        running_loss = 0.0
        
        for i, (image, latents, label, key) in enumerate(dataloader):
            image = image.to(device)
            optimizer_warmup.zero_grad()
            
            # Forward (no latents in warmup)
            output_img = reformer(image)
            
            # Reconstruction loss
            mse_loss = mse_criterion(output_img, image)
            ssim_loss = 1 - ssim_metric(output_img, image)
            loss = mse_loss + opt.lambda_ssim * ssim_loss
            
            loss.backward()
            optimizer_warmup.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader)
        print(f"Warmup Epoch [{epoch+1}/{opt.warmup_epochs}], Loss: {avg_loss:.4f}")
    
    print("\nWarmup completed!\n")
    
    # Phase 2: Convert to latent-conditioned and joint train
    print("PHASE 2: Joint training LatentNet + Reformer...")
    print("-" * 60)
    
    # Create new latent-conditioned reformer and initialize from warmup
    reformer_conditioned = LatentConditionedReformer(in_channels=3, latent_channels=16).to(device)
    
    # Copy weights from basic reformer where possible
    reformer_state = reformer.state_dict()
    reformer_conditioned_state = reformer_conditioned.state_dict()
    
    # Transfer encoder and decoder weights
    for key in reformer_state.keys():
        if key in reformer_conditioned_state:
            if reformer_state[key].shape == reformer_conditioned_state[key].shape:
                reformer_conditioned_state[key] = reformer_state[key]
    
    reformer_conditioned.load_state_dict(reformer_conditioned_state, strict=False)
    
    # Create LatentNet
    latent_net = LatentNet(latent_dim=16, max_length=15,
                          bottleneck_H=opt.imgH, bottleneck_W=opt.imgW).to(device)
    
    # Joint optimizer
    optimizer = optim.Adam(
        list(reformer_conditioned.parameters()) + list(latent_net.parameters()),
        lr=opt.lr
    )
    
    ctc_criterion = nn.CTCLoss(zero_infinity=True).to(device)
    
    # Joint training
    num_joint_epochs = opt.num_epochs - opt.warmup_epochs
    
    for epoch in range(num_joint_epochs):
        reformer_conditioned.train()
        latent_net.train()
        running_loss = 0.0
        sample_idx = random.randint(0, len(dataset) - 1)
        
        for i, (image, latents, label, key) in enumerate(dataloader):
            image = image.to(device)
            latents = latents.to(device)
            
            if latents.dim() == 2:
                latents = latents.unsqueeze(0)
            
            optimizer.zero_grad()
            
            # Forward
            latents_expanded = latent_net(latents)
            output_img = reformer_conditioned(image, latents_expanded)
            
            # Reconstruction loss
            mse_loss = mse_criterion(output_img, image)
            ssim_loss = 1 - ssim_metric(output_img, image)
            recon_loss = mse_loss + opt.lambda_ssim * ssim_loss
            
            # CTC loss
            length_for_pred = torch.IntTensor([opt.batch_max_length]).to(device)
            text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)
            
            with torch.no_grad():
                preds = model(output_img, text_for_pred)
            
            ctc_loss_val = torch.tensor(0.0, device=device)
            if 'CTC' in opt.Prediction:
                text_for_loss, length_for_loss = converter.encode([label[0]], 
                                                                   batch_max_length=opt.batch_max_length)
                preds_size = torch.IntTensor([preds.size(1)]).to(device)
                ctc_loss_val = ctc_criterion(preds.log_softmax(2).permute(1, 0, 2),
                                            text_for_loss, preds_size, length_for_loss)
            
            # Total loss
            loss = opt.alpha * recon_loss + opt.beta * ctc_loss_val
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if i == sample_idx:
                sample_orig = image[0].cpu()
                sample_recon = output_img[0].detach().cpu()
                sample_label = label[0]
        
        avg_loss = running_loss / len(dataloader)
        print(f"Joint Epoch [{epoch+1}/{num_joint_epochs}], "
              f"Loss: {avg_loss:.4f}, CTC: {ctc_loss_val.item():.4f}")
        
        # Visualize
        if epoch % max(1, num_joint_epochs // 5) == 0:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(tensor_to_numpy(sample_orig))
            axes[0].set_title(f'Original\nLabel: {sample_label}')
            axes[0].axis('off')
            axes[1].imshow(tensor_to_numpy(sample_recon))
            axes[1].set_title('Reconstruction')
            axes[1].axis('off')
            plt.suptitle(f'Joint Training Epoch {epoch+1}')
            plt.savefig(os.path.join(output_dir, f'joint_epoch_{epoch+1}.png'))
            plt.close()
    
    # Save models
    torch.save(reformer_conditioned.state_dict(), './pipeline4_reformer.pth')
    torch.save(latent_net.state_dict(), './pipeline4_latentnet.pth')
    print("\nJoint training completed. Models saved.\n")
    return reformer_conditioned, latent_net


def pipeline4_test_npz_latent(opt):
    """Test Pipeline 4 models"""
    print("\n" + "="*60)
    print("PIPELINE 4 - TESTING: npz → LatentNet + Reformer → STR")
    print("="*60 + "\n")
    
    # Setup converter and model
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    
    model = Model(opt).to(opt.device)
    model.load_state_dict(torch.load(opt.str_model, map_location=opt.device), strict=False)
    model.eval()
    
    # Load reformer and latent_net
    reformer = LatentConditionedReformer(in_channels=3, latent_channels=16).to(opt.device)
    reformer.load_state_dict(torch.load('./pipeline4_reformer.pth', map_location=opt.device))
    reformer.eval()
    
    latent_net = LatentNet(latent_dim=16, max_length=15,
                          bottleneck_H=opt.imgH, bottleneck_W=opt.imgW).to(opt.device)
    latent_net.load_state_dict(torch.load('./pipeline4_latentnet.pth', map_location=opt.device))
    latent_net.eval()
    
    # Prepare output
    str_name = opt.str_model.split('/')[-1].split('-')[0]
    test_output_path = os.path.join(opt.output, 'pipeline4_npz_latent', str_name)
    attack_success_result = os.path.join(test_output_path, 'attack_success_result.txt')
    save_success_adv = os.path.join(test_output_path, 'attack-success-adv')
    makedirs(test_output_path)
    makedirs(save_success_adv)
    
    # Load adversarial data
    dataset = NPZDataset(opt.npz_adv_path)
    
    result = dict()
    print(f"Testing {len(dataset)} adversarial samples...\n")
    
    for i in range(len(dataset)):
        image, latents, label, key = dataset[i]
        image = image.unsqueeze(0).to(opt.device)
        latents = latents.to(opt.device)
        
        if latents.dim() == 2:
            latents = latents.unsqueeze(0)
        
        # Reconstruct
        with torch.no_grad():
            latents_expanded = latent_net(latents)
            reconstructed_img = reformer(image, latents_expanded)
        
        # Predict
        length_for_pred = torch.IntTensor([opt.batch_max_length]).to(opt.device)
        text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(opt.device)
        
        with torch.no_grad():
            if 'CTC' in opt.Prediction:
                preds = model(reconstructed_img, text_for_pred).log_softmax(2)
                preds_size = torch.IntTensor([preds.size(1)])
                _, preds_index = preds.permute(1, 0, 2).max(2)
                preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
                preds_output = converter.decode(preds_index.data, preds_size)[0]
            else:
                preds = model(reconstructed_img, text_for_pred, is_train=False)
                _, preds_index = preds.max(2)
                preds_output = converter.decode(preds_index, length_for_pred)[0]
                preds_output = preds_output[:preds_output.find('[s]')]
        
        result[i] = f'{key}:{label}--->{preds_output}\n'
        
        if i < opt.visualize_samples:
            print(f"Sample {i+1}: Label='{label}', Pred='{preds_output}'")
            
            if i < 3:
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                axes[0].imshow(tensor_to_numpy(image[0]))
                axes[0].set_title(f'Original Adv\nLabel: {label}')
                axes[0].axis('off')
                axes[1].imshow(tensor_to_numpy(reconstructed_img[0]))
                axes[1].set_title(f'Reconstruction\nPred: {preds_output}')
                axes[1].axis('off')
                plt.tight_layout()
                plt.show()
    
    # Write results
    result = sorted(result.items(), key=lambda x: x[0])
    with open(attack_success_result, 'w+') as f:
        for _, line in result:
            f.write(line)
    
    # Evaluate
    evaluate_predictions(attack_success_result, save_success_adv)
if __name__ == "__main__":
    main()