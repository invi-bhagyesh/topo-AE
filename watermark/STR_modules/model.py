import os
import sys
# Add project root to sys.path (two levels up from the current file)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import torch.nn as nn
import torch
from .transformation import TPS_SpatialTransformerNetwork
from .feature_extraction import VGG_FeatureExtractor, ResNet_FeatureExtractor
from .sequence_modeling import BidirectionalLSTM
from .prediction import Attention
from src.models.approx_based import TopologicallyRegularizedAutoencoder
from src.models.submodules import DeepAE
from src.evaluation.utils import get_space
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import subprocess
from glob import glob
from PIL import Image
import torchvision.utils as vutils
import os

class ImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.paths = sorted(glob(f"{root}/*.png"))
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


#####################################################################
# Reformer + STR models

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
    


class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        
        self.stages = {
            'Trans': opt.Transformation,
            'Feat': opt.FeatureExtraction,
            'Seq': opt.SequenceModeling,
            'Pred': opt.Prediction
        }

        print(f"[INFO] Initializing Model with stages: {self.stages}")

        """ Transformation : output is rectified image [batch_size x I_channel_num x I_r_height x I_r_width] """
        if opt.Transformation == 'TPS':
            print("[Stage: Trans] Using TPS_SpatialTransformerNetwork")
            print("[INFO] Initializing TPS_SpatialTransformerNetwork with parameters:")
            print(f"   F (num_fiducial): {opt.num_fiducial}")
            print(f"   I_size: {(opt.imgH, opt.imgW)}")
            print(f"   I_r_size: {(opt.imgH, opt.imgW)}")
            print(f"   I_channel_num: {opt.input_channel}")
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial,
                I_size=(opt.imgH, opt.imgW),
                I_r_size=(opt.imgH, opt.imgW),
                I_channel_num=opt.input_channel
            )
        else:
            print("[Stage: Trans] No Transformation module specified")

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel) # 512x1x24
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size), 
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))   # hidden_size = 256, output is [batch_size x T x output_size]
            self.SequenceModeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)  # batch_size x num_steps x num_classes
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input) # bx512x1x24
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]=bx24x512x1
        visual_feature = visual_feature.squeeze(3) # [b, w, c]=bx24x512

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature) # [b, w, hidden]=bx24x256
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous()) #[b, w, class]=bx24x63
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return prediction


class FullModel(nn.Module):
    def __init__(self, opt, reformer_ckpt_path=None,
                 topo_ckpt_path="/kaggle/input/fawa_topo_ae/pytorch/default/2/model_state.pth",
                  data_dir="/kaggle/input/test-adv-splitted/test_original", 
                  device="cuda"):
        super().__init__()
        self.opt = opt
        self.data_dir = data_dir

        # Init topo model
        self.topo_model = TopologicallyRegularizedAutoencoder(
            ae_kwargs={'input_dims': [3, 28, 44]},
            autoencoder_model="DeepAE",
            lam=1.6280214927932581,
            toposig_kwargs={"match_edges": "symmetric"}
        )
        if topo_ckpt_path is not None:
            state_dict = torch.load(topo_ckpt_path, map_location=device)
            self.topo_model.load_state_dict(state_dict)
            print(f"Loaded TopoModel weights from {topo_ckpt_path}")

        self.topo_model.eval()
        if device == "cuda":
            self.topo_model = self.topo_model.cuda()

        # Init Latent Reformer
        self.latent_reformer = LatentReformer(in_channels=opt.input_channel)
        if reformer_ckpt_path is not None:
            ckpt = torch.load(reformer_ckpt_path, map_location="cpu")
            self.latent_reformer.load_state_dict(ckpt, strict=True)
            print(f"Loaded LatentReformer weights from {reformer_ckpt_path}")
        # Main OCR Model
        self.ocr_model = Model(opt)

    def forward(self, images, text, is_train=True):
        #  Split with script.py
        subprocess.run([
            "python3", "scripts/split/script_test.py",
            "--mode", "split",
            "--split_input", self.data_dir,
            "--split_output", "characters",
            "--padding", "5",
            "--char_size", "64"
        ])
        transform = transforms.Compose([
            transforms.Resize((self.opt.imgH, self.opt.imgW)),
            transforms.ToTensor()
        ])
        dataset = ImageFolder(root="characters", transform=transform)
        dataloader = DataLoader(dataset, batch_size=self.opt.batch_size, shuffle=False)



        os.makedirs("characters_recon", exist_ok=True)

        with torch.no_grad():
            for batch_idx, (char_imgs, _) in enumerate(dataloader):
                char_imgs = char_imgs.to(next(self.topo_model.parameters()).device)
                latent = self.topo_model.encode(char_imgs)
                reconstructed = self.topo_model.decode(latent)

                for j, img_tensor in enumerate(reconstructed):
                    save_path = f"characters_recon/reconst_{batch_idx}_{j}.png"
                    vutils.save_image(img_tensor.cpu(), save_path)
        
        # Step 2: Combine with script.py
        subprocess.run([
            "python3", "scripts/split/script_test.py",
            "--mode", "combine",
            "--combine_input", "characters_recon",
            "--combine_output", "reconstructed",
            "--original_input", self.data_dir
        ])

        
        # # Step 1: Pass through Latent Reformer
        # input = self.latent_reformer(reconstructed)

        # # Step 2: Pass through OCR model
        # prediction = self.ocr_model(input, text, is_train)
        # return prediction

        # Step 3: Load combined images from reconstructed/
        dataset_recon = ImageFolder(root="reconstructed", transform=transform)
        dataloader_recon = DataLoader(dataset_recon, batch_size=self.opt.batch_size, shuffle=False)

        all_predictions = []
        device = next(self.latent_reformer.parameters()).device
        for images, _ in dataloader_recon:
            images = images.to(device)
            reformer_out = self.latent_reformer(images)
            preds = self.ocr_model(reformer_out, text, is_train)
            all_predictions.append(preds)

        # Concatenate into one tensor like earlier
        prediction = torch.cat(all_predictions, dim=0)
        return prediction