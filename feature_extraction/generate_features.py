
###
# This file will:
# 1. Generate and save features in a given folder
###

import glob
# from msilib import sequence
import numpy as np
import torch
import argparse
import random
from torch.utils.data import Dataset, DataLoader
import os
import h5py

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils_py import *
seed = 0
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Python RNG
np.random.seed(seed)
random.seed(seed)

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
class Ds_loader(Dataset):
    def __init__(self, arr, transform):
        self.x = arr
        self.transform = transform

    def __getitem__(self, idx):
        x = self.transform(self.x[idx])
        return x

    def __len__(self):
        return self.x.shape[0]

def get_arr_from_hdf(file):
    with h5py.File(file, 'r') as f:
        arr = f['stimuli'][:]   # (N, H, W, C)
    # arr = np.transpose(arr, (0, 3, 1, 2))   # (N, C, H, W)
    return arr


def get_activations_and_save(args, model_name, data_dir, save_dir, dim_rd,
                            pca_ratio, tr_frms = 30):
    """This function generates features and save them in a specified directory.
    Parameters
    ----------
    model_name : name of the pretrained model.
    split : str
        train or test.
    data_dir : str
        data directory
    save_dir : str
        save path to extracted features.
    dim_rd : str
        if reduce the dimension of the extracted features.
    pca_ratio: float
        ratio of pca component only if dim_rd='pca'
    tr_frms : int
        how many frames per TR.
    """

    model = load_model(model_name, args)
    transform = image_transform(model_name, model=model)
    layer_3d = 0
    for split in ['train', 'test']:
        j = 0
        file_path_list = glob.glob(os.path.join(data_dir, 'stimuli', '%s*.hdf'%split))
        print(file_path_list)
        file_path_list.sort()

        for file_num, file_path in enumerate(file_path_list):
            print(file_path)
            arr = get_arr_from_hdf(file_path)
            save_dir_notrd = os.path.join(save_dir, 'not_rd')
            if not os.path.exists(save_dir_notrd):
                os.makedirs(save_dir_notrd)
            file_out = os.path.join(save_dir_notrd, '%s_fea.hdf'%split)
            data_loader = DataLoader(Ds_loader(arr, transform), 
                                batch_size=tr_frms, shuffle=False, drop_last=False)
            # print(arr)
            for tr_num, imgs in enumerate(data_loader):
                imgs = imgs.float().cuda()
                with torch.no_grad():
                    out_fea_list = feature_extraction(model_name, model, imgs)
                    print(out_fea_list)
                    exit()
                for fea_num, feat in enumerate(out_fea_list):
                    # average feature maps on the frames during one TR
                    feat_tmp = np.mean(feat.data.cpu().numpy(), axis=0, keepdims=True)
                    if split == 'train' and j == 0:
                        if len(feat_tmp.shape) == 2 and layer_3d == 0:
                            layer_3d = fea_num
                    if file_num == 0 and tr_num == 0:
                        mode = 'w' if fea_num == 0 else 'a'
                        with h5py.File(file_out, mode) as f:
                            print('create dataset of shape=', 
                                    arr.shape[0]/tr_frms*len(file_path_list), 
                                    feat_tmp.ravel().shape[-1])
                            f.create_dataset("layer_%s"%str(fea_num+1).zfill(2),
                                            shape = (arr.shape[0]/tr_frms*len(file_path_list), 
                                                    feat_tmp.ravel().shape[-1]))
                            f["layer_%s"%str(fea_num+1).zfill(2)][j] = feat_tmp.ravel()
                    else:
                        with h5py.File(file_out, 'a') as f:
                            f["layer_%s"%str(fea_num+1).zfill(2)][j] = feat_tmp.ravel()
                j += 1
    
    save_dir_notrd = os.path.join(save_dir, 'not_rd')
    file_train_out = os.path.join(save_dir_notrd, 'train_fea.hdf')
    file_test_out = os.path.join(save_dir_notrd, 'test_fea.hdf')
    print('Features before layer-%d can be considered PCA'%layer_3d)
    if dim_rd == 'pca':
        if not os.path.exists(os.path.join(save_dir, dim_rd)):
            os.makedirs(os.path.join(save_dir, dim_rd))
        for layer in range(len(out_fea_list)):
            with h5py.File(file_train_out, 'r') as f:
                fea_tot_layer_train = f["layer_%s"%str(layer+1).zfill(2)][:]
            with h5py.File(file_test_out, 'r') as f:
                fea_tot_layer_test = f["layer_%s"%str(layer+1).zfill(2)][:]
            print('not_rd', layer, fea_tot_layer_test.shape[-1])
            if layer < layer_3d:
                fea_tot_layer_train = StandardScaler().fit_transform(fea_tot_layer_train)
                fea_tot_layer_test = StandardScaler().fit_transform(fea_tot_layer_test)
                ipca = PCA(n_components=pca_ratio, svd_solver='full')
                ipca.fit(fea_tot_layer_train)
                fea_tot_layer_train = ipca.transform(fea_tot_layer_train)
                fea_tot_layer_test = ipca.transform(fea_tot_layer_test)
                print(dim_rd, layer, ipca.n_components_)
            mode = 'w' if layer == 0 else 'a'
            with h5py.File(os.path.join(save_dir, dim_rd, 'train_fea.hdf'), mode) as f:
                f.create_dataset("layer_%s"%str(layer+1).zfill(2),
                                data=fea_tot_layer_train)
            with h5py.File(os.path.join(save_dir, dim_rd, 'test_fea.hdf'), mode) as f:
                f.create_dataset("layer_%s"%str(layer+1).zfill(2),
                                data=fea_tot_layer_test)


def get_activations_and_save_text(args, model_name, data_dir, save_dir, dim_rd,
                            pca_ratio, tr_frms = 30):
    """This function generates features and save them in a specified directory.
    Parameters
    ----------
    model_name : name of the pretrained model.
    split : str
        train or test.
    data_dir : str
        data directory
    save_dir : str
        save path to extracted features.
    dim_rd : str
        if reduce the dimension of the extracted features.
    pca_ratio: float
        ratio of pca component only if dim_rd='pca'
    tr_frms : int
        how many frames per TR.
    """

    model = load_model(model_name, args)
    transform = text_transform()

    feature_list = []
    # stimuli_384sentences_dereferencedpronouns.txt
    with open('/home/haoyu_lu/NATURE2022/text/data/stimuli_384sentences_dereferencedpronouns.txt') as f:
        for line in f.readlines():
            # print(line)
            word = line
            print(word)
            word = transform.transform(word)
            # print(word)
            with torch.no_grad():
                word = model(word[0].cuda(), word[1].cuda())
            feature_list.append(word.unsqueeze(0))
    
    feature = torch.cat(feature_list)
    print(feature.shape) # torch.Size([1705, 12, 768])
    torch.save(feature, 'stimuli_384sentences_dereferencedpronouns_wenlan_1400w_coco_feature.pth')




def main():
    parser = argparse.ArgumentParser(description='Feature Extraction')
    parser.add_argument('-rp','--root_path', help='root path', 
            default = "./", type=str)
    parser.add_argument('-ddir','--data_dir', help='data directory',
            default = "./datasets/", type=str)
    parser.add_argument('--model_path', help='model directory',
                        default = "", type=str)
    parser.add_argument('-fname','--feature_name', help='feature name, for example, alexnet',
            default = 'wenlan', type=str)
    parser.add_argument('-dim_rd','--dim_rd', help='feature dimension reduction', default = 'not_rd', type=str)
    parser.add_argument('-pca_ratio','--pca_ratio', help='pca component ratio', default = 0.999, type=float)
    parser.add_argument('-tr_frm','--tr_frames', help='frame num per TR', default = 30, type=int)
    parser.add_argument('-gpu', '--gpu_id', help='gpu id', default='3', type=str)
    # parser.add_argument('--cfg_file', type=str, default='cfg/moco_box.yml')
    args = vars(parser.parse_args())

    os.environ['CUDA_VISIBLE_DEVICES']=args['gpu_id']
    save_dir = os.path.join(args['root_path'], 'feature', args['feature_name'], '10_19_15:33:07')
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    print("-------------Saving activations ----------------------------")
    print(args)
    if args['feature_name'] == 'bert' or args['feature_name'] == 'wenlan_text':
        get_activations_and_save_text(args,
                            args['feature_name'], 
                            args['data_dir'], save_dir, 
                            args['dim_rd'], pca_ratio=args['pca_ratio'],
                            tr_frms=args['tr_frames'])
    else:
        get_activations_and_save(args,
                                args['feature_name'], 
                                args['data_dir'], save_dir, 
                                args['dim_rd'], pca_ratio=args['pca_ratio'],
                                tr_frms=args['tr_frames'])


if __name__ == "__main__":
    
    main()
