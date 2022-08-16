import torch
import torch.nn as nn
import torchvision

from .fakeTransformer import FakeTransformer
from .bert import Bert

from utils import pairLoss, alignmentLoss, attAlignmentLoss, AlignTripLoss, SimpTripLoss, NCELoss
import torch.nn.functional as F
import timm

from transformers.models.bert.modeling_bert import BertPredictionHeadTransform
from transformers import BertConfig

from .vision_transformer import vit_base_patch16_384


import numpy as np
import math

class ImgLearnableEncoder(nn.Module):
    def __init__(self, model_cfg, args):
        super(ImgLearnableEncoder, self).__init__()


        # self.backbone_vit = create_beit_model(args)
        # self.backbone_vit = timm.create_model(model_cfg.VIT, pretrained=True)
        self.backbone_vit = vit_base_patch16_384(pretrained=True)

        self.model_cfg = model_cfg
        self.learnable = nn.ModuleDict()
        
        img_encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_cfg.IMG_FEATURE_DIM, nhead=self.model_cfg.IMG_TRANSFORMER_HEAD)
        self.learnable['imgAtt'] = nn.TransformerEncoder(img_encoder_layer, num_layers=self.model_cfg.IMG_TRANSFORMER_LAYER)

        self.learnable['imgFC_vit'] = FakeTransformer(model_cfg.IMG_FEATURE_DIM_vit, model_cfg.HIDDEN_DIM_1, model_cfg.HIDDEN_DIM_2)
        self.learnable['imgFC_vit_0'] = FakeTransformer(model_cfg.IMG_FEATURE_DIM_vit, model_cfg.HIDDEN_DIM_1, model_cfg.HIDDEN_DIM_2)

        self.learnable['final_mlp'] = FakeTransformer(model_cfg.HIDDEN_DIM_2*2, model_cfg.HIDDEN_DIM_2, model_cfg.HIDDEN_DIM_2)
        self.init_param()

    def init_param(self):

        
        for name, param in self.backbone_vit.named_parameters():
            # param.requires_grad = True
            # print('5' * 100)
            # print(name)
            if 'blocks.11' in name or 'blocks.10' in name or 'blocks.9' in name or 'blocks.8' in name or 'blocks.7' in name or 'blocks.6' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
            

    def forward(self, imgFea, ):

        imgFea_vit, feature_list = self.backbone_vit.forward_features(imgFea)
        

        

        imgFea_vit = self.learnable['imgFC_vit_0'](imgFea_vit)
        feature_list.append(imgFea_vit)
        
        imgFea_vit = F.normalize(imgFea_vit, p=2, dim=-1)
        feature_list.append(imgFea_vit)
        

        return feature_list


class TextLearnableEncoder(nn.Module):
    def __init__(self, model_cfg):
        super(TextLearnableEncoder, self).__init__()

        self.backbone = Bert(model_cfg)
        self.model_cfg = model_cfg

        self.learnable = nn.ModuleDict()
        
        text_encoder_layer = nn.TransformerEncoderLayer(d_model=model_cfg.TEXT_FEATURE_DIM, nhead=model_cfg.TEXT_TRANSFORMER_HEAD)
        self.learnable['textAtt'] = nn.TransformerEncoder(text_encoder_layer, num_layers=model_cfg.TEXT_TRANSFORMER_LAYER)

        self.learnable['textFC'] = FakeTransformer(model_cfg.TEXT_FEATURE_DIM, model_cfg.HIDDEN_DIM_1, model_cfg.HIDDEN_DIM_2)

        self.init_param()

    def init_param(self):

        for name, param in self.backbone.named_parameters():
            if 'layer.11' not in name and 'layer.10' not in name and 'layer.9' not in name and 'layer.8' not in name and 'layer.7' not in name and 'layer.6' not in name: #  and 'layer.9' not in name
                param.requires_grad = False
            else:
                param.requires_grad = True


    def forward(self, textFea, maskTexts):

        textFea = self.backbone(textFea)

        return textFea





class VL_model(nn.Module):

    def __init__(self, model_cfg, args):
        super(VL_model, self).__init__()

        self.model_cfg = model_cfg

        self.learnable = nn.ModuleDict()
        self.learnable['imgencoder'] = ImgLearnableEncoder(model_cfg, args)
        self.learnable['imgencoder_mom'] = ImgLearnableEncoder(model_cfg, args)
        self.learnable['textencoder'] = TextLearnableEncoder(model_cfg)
        self.learnable['textencoder_mom'] = TextLearnableEncoder(model_cfg)



    def forward(self, imgFea, texts, maskImages, maskTexts, text_lens, image_boxs, is_training=True):

        return self.extract(imgFea, texts, maskImages, maskTexts, image_boxs)



    def extract(self, imgFea, texts, maskImages, maskTexts, image_boxs):


        imgFea, _ = self.learnable['imgencoder'](imgFea, maskImages, image_boxs) # <bsz, img_dim>
        textFea, _ = self.learnable['textencoder'](texts, maskTexts) # <bsz, img_dim>
        
        imgFea = F.normalize(imgFea, p=2, dim=-1)
        textFea = F.normalize(textFea, p=2, dim=-1)

        retrieval_feat_group = {}

        retrieval_feat_group['img_text'] = (imgFea, textFea)

        return retrieval_feat_group
