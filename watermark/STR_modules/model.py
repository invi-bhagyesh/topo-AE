import torch.nn as nn
import torch
from .transformation import TPS_SpatialTransformerNetwork
from .feature_extraction import VGG_FeatureExtractor, ResNet_FeatureExtractor
from .sequence_modeling import BidirectionalLSTM
from .prediction import Attention

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
        if 'TPS'== 'TPS':
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
    def __init__(self, opt, reformer_ckpt_path=None):
        super().__init__()
        self.opt = opt
        # Init Latent Reformer
        self.latent_reformer = LatentReformer(in_channels=opt.input_channel)
        if reformer_ckpt_path is not None:
            ckpt = torch.load(reformer_ckpt_path, map_location="cpu")
            self.latent_reformer.load_state_dict(ckpt, strict=True)
            print(f"Loaded LatentReformer weights from {reformer_ckpt_path}")
        # Main OCR Model
        self.ocr_model = Model(opt)

    def forward(self, input, text, is_train=True):
        # Step 1: Pass through Latent Reformer
        input = self.latent_reformer(input)

        # Step 2: Pass through OCR model
        prediction = self.ocr_model(input, text, is_train)
        return prediction
