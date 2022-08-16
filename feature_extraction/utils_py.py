# model loading, image transform, feature extraction of different models
from mimetypes import init
import os
import torch
import urllib
import argparse
from torchvision import transforms as trn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from alexnet import *
from models import build_network
from utils.config import cfg_from_yaml_file, cfg
from transformers import AutoTokenizer

def getLanMask(seq_lens, max_len):
    # seq_lens (bs)
    mask = torch.ones((seq_lens.size(0), max_len))  # (bs, max_len)
    idxs = torch.arange(max_len).unsqueeze(dim=0)  # (1, max_len)
    seq_lens = seq_lens.unsqueeze(-1)  # (bs, 1)
    mask = torch.where(idxs < seq_lens, mask, torch.Tensor([0.0]))
    return mask

def load_alexnet(model_checkpoints):
    """This function initializes an Alexnet and load
    its weights from a pretrained model
    ----------
    model_checkpoints : str
        model checkpoints location.

    Returns
    -------
    model
        pytorch model of alexnet
    """
    model = alexnet()

    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model

def load_wenlan(args, load=False):


    cfg_from_yaml_file('./feature_extraction/cfg/moco_box.yml', cfg)

    cfg.MODEL.TEXT_FEATURE_DIM = 768
    cfg.MODEL.IMG_SIZE = 384
    
    model = build_network(cfg.MODEL, args)
    
    if load:
        print('load wenlan feature!!')

        model_component = torch.load(args.model_path + '/mp_rank_00_model_states.pt', map_location=torch.device('cpu'))

        model_dict = model.state_dict()
        state_dict = {k:v for k,v in model_component['module'].items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    print('load wenlan down')
    return model

def load_model(model_name, args):
    if model_name == 'alexnet':
        # load Alexnet
        # Download pretrained Alexnet from:
        # https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth
        # and save in the current directory
        checkpoint_path = "./alexnet.pth"
        # if not os.path.exists(checkpoint_path):
        #     url = "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth"
        #     urllib.request.urlretrieve(url, "./alexnet.pth")
        model = load_alexnet(checkpoint_path)
        return model
    elif 'wenlan' in model_name and 'text' not in model_name:
        model = load_wenlan(args, load=True)
        return model.learnable['imgencoder']
    elif model_name == 'vit':
        model = load_wenlan(args, load=False)
        return model.learnable['imgencoder']
    elif model_name == 'wenlan_text':
        model = load_wenlan(args, load=True)
        return model.learnable['textencoder']
    elif model_name == 'bert':
        model = load_wenlan(args, load=False)
        return model.learnable['textencoder']
    else:
        model = load_wenlan(args)
        return model.learnable['imgencoder']

class text_transform(object):
    def __init__(self):
        self.text_transform = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    def transform(self, text):

        text_info = self.text_transform(text, padding='max_length', truncation=True,
                                            max_length=80, return_tensors='pt')
        
        text = text_info.input_ids.reshape(-1).unsqueeze(0)
        text_len = torch.sum(text_info.attention_mask)
        
        textMask = getLanMask(text_len.unsqueeze(0), 80)
        return text, textMask
        

def image_transform(model_name, model=None):
    if model_name == 'alexnet':
        transform = trn.Compose([
        trn.ToPILImage(),
        trn.Resize(224),
        trn.ToTensor(),
        trn.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
        return transform
    elif 'wenlan' in model_name and 'text' not in model_name:
        normalize = trn.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

        return trn.Compose([
                trn.ToPILImage(),
                trn.RandomResizedCrop(384, scale=(0.2, 1.)),
                trn.ToTensor(),
                normalize])
    elif model_name == 'bert' or model_name == 'wenlan_text':
        pass



def feature_extraction(model_name, model, image):
    if model_name == 'alexnet':
        x = model.forward(image)
        return x
    else:
        x = model(image)
        return x

