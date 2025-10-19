import random
import sys 
import os
sys.path.append('/Users/invi/Desktop/TopoReformer/TopoReformer/watermark')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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


import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from nltk.metrics import edit_distance

def tensor_to_numpy(img_tensor):
    img = img_tensor.cpu().clamp(0,1).detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    return img

def makedirs(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            print(f'Cannot create dirs: {path}')
            sys.exit(0)

def process_line(line):
    adv_img_path, recog_result = line.split(':')
    label, adv_preds = recog_result.split('--->')
    adv_preds = adv_preds.strip('\n')
    return adv_preds, label, adv_img_path

def test(opt, reformer, latent_net):
    # Setup converter and recognition model
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    model = Model(opt).to(opt.device)
    print(f'Loading STR model from "{opt.str_model}" as the target model!')
    model.load_state_dict(torch.load(opt.str_model, map_location=opt.device), strict=False)
    model.eval()
    reformer.eval()
    latent_net.eval()
    
    makedirs(opt.output)
    str_name = opt.str_model.split('/')[-1].split('-')[0]
    test_output_path = os.path.join(opt.output, opt.attack_name, str_name)
    attack_success_result = os.path.join(test_output_path, 'attack_success_result.txt')
    save_success_adv = os.path.join(test_output_path, 'attack-success-adv')
    makedirs(test_output_path)
    makedirs(save_success_adv)
    
    data = np.load('/kaggle/input/ocr-topo/filtered_combined_adv_2.npz', allow_pickle=True)
    keys = data.files
    result = dict()
    
    for i, key in enumerate(keys):
        sample = data[key].item()
        image_np = sample['image']
        image_tensor = torch.from_numpy(image_np).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(opt.device)  # B=1, C, H, W
    
        latents_np = sample['latents']  # Should already be [N, 16]
        latents_tensor = torch.from_numpy(latents_np).to(opt.device)
    
        # Ensure latents_tensor has 3 dims (B, N, 16)
        if latents_tensor.dim() == 2:
            latents_tensor = latents_tensor.unsqueeze(0)
        elif latents_tensor.dim() == 1:
            latents_tensor = latents_tensor.unsqueeze(0).unsqueeze(0)
    
        try:
            latents_expanded = latent_net(latents_tensor)
        except Exception as e:
            print(f"Skipping sample '{key}' due to error in latent_net: {e}")
            continue
    
        reconstructed_img = reformer(image_tensor, latents_expanded)
    
        length_for_pred = torch.IntTensor([opt.batch_max_length]).to(opt.device)
        text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(opt.device)
    
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
    
        result[i] = f'{key}:{sample["word_label"]}--->{preds_output}\n'
    
        if i < 10:
            orig_np = tensor_to_numpy(image_tensor[0])
            recon_np = tensor_to_numpy(reconstructed_img[0])
            plt.figure(figsize=(8,4))
            plt.subplot(1,2,1)
            plt.imshow(orig_np)
            plt.title(f'Original Adv Image\nLabel: {sample["word_label"]}')
            plt.axis('off')
            plt.subplot(1,2,2)
            plt.imshow(recon_np)
            plt.title(f'Reconstruction\nPred: {preds_output}')
            plt.axis('off')
            plt.show()
    
    result = sorted(result.items(), key=lambda x: x[0])
    with open(attack_success_result, 'w+') as f:
        for _, line in result:
            f.write(line)
    
    # Metrics calculation
    with open(attack_success_result, 'r') as f:
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
            # shutil.copy(adv_img_path, save_success_adv)  # Uncomment if paths exist
            ED_sum += edit_distance(label, adv_preds)
    
    char_level_accuracy = correct_chars / total_gt_chars if total_gt_chars > 0 else 0
    char_level_precision = correct_chars / total_pred_chars if total_pred_chars > 0 else 0
    char_level_recall = correct_chars / total_gt_chars if total_gt_chars > 0 else 0
    
    print(f'Character-level Accuracy: {char_level_accuracy * 100:.2f} %')
    print(f'Character-level Precision: {char_level_precision * 100:.2f} %')
    print(f'Character-level Recall: {char_level_recall * 100:.2f} %')
    print(f'Word-level Attack Success Rate: {attack_success_num / len(keys) * 100:.2f} %')
    if attack_success_num != 0:
        ED_num_avr = ED_sum / attack_success_num
        print(f'Average Edit_distance per word: {ED_num_avr:.2f}')

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


class test_adv_dataset(Dataset):
    def __init__(self, height, width, img_path):      
        self.height = height
        self.width = width
        self.img_path = img_path
        self.dataset = []
        
        img = [] 
        for i,j,k in os.walk(self.img_path):            
            for file in k:
                file_name = os.path.join(i ,file)
                img.append(file_name)
        self.total_img_name = img
        
        for img_name in self.total_img_name:
            base_name = os.path.basename(img_name)
            parts = base_name.rsplit('_', 2)
            if len(parts) == 3:
                img_index, label, img_adv = parts
                img_adv = img_adv.split('.')  # split extension
                index_or_advlogo = img_adv[0]
                self.dataset.append([img_name, label, img_index, index_or_advlogo])
            else:
                print(f"Skipping unexpected filename format: {base_name}")
        self.dataset = sorted(self.dataset)

    def __getitem__(self, index):
        img_name, label, img_index, index_or_advlogo = self.dataset[index]        
        IMG = cv2.imread(img_name) 
        IMG = cv2.resize(IMG, (self.width, self.height))
        
        # binarization processing
        gray = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img_b = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        img_b = torch.FloatTensor(img_b)
        img_b = img_b / 255 # normalization to [0,1]
        img_b = img_b.permute(2,0,1) # [C, H, W]

        img = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
        img = torch.FloatTensor(img)
        img = img /255 # normalization to [0,1]
        img = img.permute(2,0,1) # [C, H, W]

        return img_b, img, label, img_index, index_or_advlogo, img_name

    def __len__(self):
        return len(self.dataset)

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

    reformer = None
    if opt.use_reformer:
        reformer = LatentReformer(in_channels=3).to(opt.device)
        reformer_path = './reformer.pth'
        if os.path.exists(reformer_path):
            reformer.load_state_dict(torch.load(reformer_path, map_location=opt.device))
            reformer.eval()
        else:
            print(f"Reformer model not found at {reformer_path}. Proceeding without reformer.")
            reformer = None

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
        

        if opt.use_reformer and reformer is not None:
            reconstructed_img = reformer(adv_img)
        else:
            reconstructed_img = adv_img
        
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
    """
    Options for pipeline and model configuration.
    pipeline_mode:
        1: Pipeline 1 (wmadv → STR): Directly test adversarial images using STR model.
        2: Pipeline 2 (npz → Reformer → STR): Train and test Reformer on npz data, then test STR.
        3: Pipeline 3 (npz → LatentNet + Reformer → STR): Train both LatentNet and Reformer, then test STR.
        4: Pipeline 4 (warmup): Warmup train Reformer, then joint fine-tune LatentNet+Reformer, then test STR.
    """
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
        self.str_model = '/root/.cache/huggingface/hub/datasets--invi-bhagyesh--ocr/snapshots/1b56ad344b98c7f0aed696031cda925bd553a3c5/models/RARE-TPS-VGG-BiLSTM-Attn-sensitive.pth'
        self.Transformation = 'TPS'
        self.FeatureExtraction = 'VGG'
        self.SequenceModeling = 'BiLSTM'
        self.Prediction = 'Attn'
        self.num_fiducial = 20
        self.input_channel = 3
        self.output_channel = 512
        self.hidden_size = 256
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_reformer = True
        self.pipeline_mode = 1  # See docstring above for pipeline modes
def evaluate_predictions(result_file, save_success_adv):
    """Shared evaluation logic for all pipelines."""
    import shutil
    from nltk.metrics import edit_distance
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
    print(f'Character-level Accuracy: {char_level_accuracy * 100:.2f} %')
    print(f'Character-level Precision: {char_level_precision * 100:.2f} %')
    print(f'Character-level Recall: {char_level_recall * 100:.2f} %')
    print(f'Word-level Attack Success Rate: {attack_success_num / len(alladv) * 100:.2f} %')
    if attack_success_num != 0:
        ED_num_avr = ED_sum / attack_success_num
        print(f'Average Edit_distance per word: {ED_num_avr:.2f}')


# Placeholder for pipeline 1: wmadv → STR
def test_wmadv(opt):
    print("Running Pipeline 1: wmadv → STR")
    # load dataset, model, evaluate directly
    # compute metrics using evaluate_predictions()
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
    # Dataset and dataloader
    dataset = test_adv_dataset(opt.imgH, opt.imgW, opt.adv_img)
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
        label = data[2]
        adv_index = data[3][0]
        adv_path = data[5][0]
        length_for_pred = torch.IntTensor([opt.batch_max_length] * opt.batch_size).to(opt.device)
        text_for_pred = torch.LongTensor(opt.batch_size, opt.batch_max_length + 1).fill_(0).to(opt.device)
        if 'CTC' in opt.Prediction:
            preds = model(adv_img, text_for_pred).log_softmax(2)
            preds_size = torch.IntTensor([preds.size(1)] * opt.batch_size)
            _, preds_index = preds.permute(1, 0, 2).max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            preds_output = converter.decode(preds_index.data, preds_size)
            preds_output = preds_output[0]
            result[adv_index] = f'{adv_path}:{label[0]}--->{preds_output}\n'
        else:  # Attention
            preds = model(adv_img, text_for_pred, is_train=False)
            _, preds_index = preds.max(2)
            preds_output = converter.decode(preds_index, length_for_pred)
            preds_output = preds_output[0]
            preds_output = preds_output[:preds_output.find('[s]')]
            result[adv_index] = f'{adv_path}:{label[0]}--->{preds_output}\n'
    result = sorted(result.items(), key=lambda x: x[0])
    with open(attack_success_result, 'w+') as f:
        for item in result:
            f.write(item[1])
    evaluate_predictions(attack_success_result, save_success_adv)

# Placeholder for pipeline 2: npz → Reformer → STR
def test_npz_reformer(opt):
    print("Running Pipeline 2: npz → Reformer → STR")
    # load npz adv data, reformer, and model; run inference; evaluate
    # For demonstration, just print; replace with actual logic as needed
    # Prepare output directories
    makedirs(opt.output)
    str_name = opt.str_model.split('/')[-1].split('-')[0]
    test_output_path = os.path.join(opt.output, opt.attack_name, str_name)
    attack_success_result = os.path.join(test_output_path, 'attack_success_result.txt')
    save_success_adv = os.path.join(test_output_path, 'attack-success-adv')
    makedirs(test_output_path)
    makedirs(save_success_adv)
    # ... actual inference logic here ...
    print("Inference with Reformer only (placeholder).")
    # Write dummy result file for evaluation
    with open(attack_success_result, 'w+') as f:
        pass
    evaluate_predictions(attack_success_result, save_success_adv)

# Placeholder for pipeline 3/4: npz → LatentNet + Reformer → STR
def test_npz_latent(opt):
    print("Running Pipeline 3 or 4: npz → LatentNet + Reformer → STR")
    # similar to test_npz_reformer but includes latent_net forward pass
    makedirs(opt.output)
    str_name = opt.str_model.split('/')[-1].split('-')[0]
    test_output_path = os.path.join(opt.output, opt.attack_name, str_name)
    attack_success_result = os.path.join(test_output_path, 'attack_success_result.txt')
    save_success_adv = os.path.join(test_output_path, 'attack-success-adv')
    makedirs(test_output_path)
    makedirs(save_success_adv)
    # ... actual inference logic here ...
    print("Inference with LatentNet + Reformer (placeholder).")
    # Write dummy result file for evaluation
    with open(attack_success_result, 'w+') as f:
        pass
    evaluate_predictions(attack_success_result, save_success_adv)

# Placeholder for pipeline 4 warmup
def train_reformer_warmup(opt, model, converter):
    print("Warmup training Reformer only (Pipeline 4)")
    # run limited epochs of train_reformer
    return train_reformer(opt, model, converter, num_epochs=2)

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
    
# LatentNet: expands 16-dim latent sequences to spatial bottleneck feature maps
class LatentNet(nn.Module):
    def __init__(self, latent_dim=16, max_length=15, bottleneck_H=32, bottleneck_W=100):
        super().__init__()
        self.latent_dim = latent_dim  # should match input latent dim (16)
        self.max_length = max_length  # max sequence length (padding length)
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
        if N < self.max_length:
            pad_size = self.max_length - N
            padding = torch.zeros(B, pad_size, D, device=latents.device, dtype=latents.dtype)
            latents = torch.cat([latents, padding], dim=1)
        x = latents.reshape(B, -1)  # reshape sequence & features for FC input
        x = self.fc(x)
        x = x.view(B, self.latent_dim, self.bottleneck_H, self.bottleneck_W)  # spatial reshaping
        return x


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


if __name__ == "__main__":
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Model and converter setup (shared)
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    model = Model(opt).to(opt.device)

    # Pipeline selection
    if opt.pipeline_mode == 1:
        # Pipeline 1: wmadv → STR
        test_wmadv(opt)
    elif opt.pipeline_mode == 2:
        # Pipeline 2: npz → Reformer → STR
        # Train and test Reformer
        print("Training Reformer (Pipeline 2)")
        train_reformer(opt, model, converter)
        test_npz_reformer(opt)
    elif opt.pipeline_mode == 3:
        # Pipeline 3: npz → LatentNet + Reformer → STR
        print("Training LatentNet + Reformer (Pipeline 3)")
        # train both
        train_reformer(opt, model, converter)
        test_npz_latent(opt)
    elif opt.pipeline_mode == 4:
        # Pipeline 4: warmup
        print("Warmup training Reformer (Pipeline 4)")
        train_reformer_warmup(opt, model, converter)
        print("Joint fine-tuning LatentNet + Reformer (Pipeline 4)")
        train_reformer(opt, model, converter)
        test_npz_latent(opt)
    else:
        print(f"Unknown pipeline_mode {opt.pipeline_mode}. Please select 1-4.")
