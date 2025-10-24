import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from nltk.metrics import edit_distance

class Opt:
    def __init__(self):
        self.output = 'res-BlackModelTest/up5a'
        self.attack_name = 'baseline_attack'  # set this accordingly
        self.adv_img = '/kaggle/input/ocr-topo/filtered_combined_train2.npz'  # set your adversarial image path here
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




test(opt,reformer,latent_net)