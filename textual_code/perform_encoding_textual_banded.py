# -*- coding: utf-8 -*-
# Experiments on the textual dataset:
# Using features of both models and split their contribution
import numpy as np
import os
import argparse
import csv
import time
import pickle
import h5py
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys

sys.path.append('/home/haoyu_lu/NATURE2022/encoding_wenlan/textual_code')
import glob
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from himalaya.backend import set_backend
from himalaya.kernel_ridge import MultipleKernelRidgeCV, Kernelizer, ColumnKernelizer
from himalaya.scoring import r2_score_split
import scipy.io as sio
import torch

def load_data(root_path, model_list, fea_layer, subject_id, exp):
    '''
    subject_id: [P01,M02,M04,M07,M08,M09,M14,M15]
    exp: examples_180concepts_sentences or examples_384sentences
    '''
    fmri_path = os.path.join(root_path, subject_id, exp+'.mat')
    fmri_file = sio.loadmat(fmri_path)
    fmri = fmri_file['examples']    #(sample_num), (voxel_num)

    # fea_path = os.path.join(root_path, 'features')
    fea_path = '/home/haoyu_lu/NATURE2022/encoding_wenlan/text_feature'
    if '180' in exp:
        prefix = 'stimuli_180concepts'
    elif '384' in exp:
        prefix = 'stimuli_384sentences_dereferencedpronouns'

    fea = []
    dim_list = []
    if fea_layer  <= 0:
        print('fea_layer should be positive!')
        return
    else:
        for model in model_list:
            # if 'wenlan' in model: 
            #     fea_pth_file = os.path.join(fea_path, "%s_%s_feature.pth"%(prefix, 'wenlan_1400w_coco'))
            # else:
            fea_pth_file = os.path.join(fea_path, "%s_%s_feature.pth"%(prefix, model))
            fea_3d = torch.load(fea_pth_file).detach().cpu().numpy()
            fea_tmp = fea_3d[:, fea_layer-1, :]
            fea.append(fea_tmp)
            dim_list.append(fea_tmp.shape[-1])
    print('dim_list', dim_list)
    fea = np.hstack(fea)
    return fea, fmri, dim_list

def write_csv(data, field_names, csv_path):
    if not os.path.isfile(csv_path):
        with open(csv_path, "w") as csv_file:
            writer = csv.DictWriter(
                csv_file, fieldnames=field_names)
            writer.writeheader()
    with open(csv_path, "a") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=field_names)
        writer.writerow(data)


def main():
    parser = argparse.ArgumentParser(description='Encoding model analysis for the textual dataset')
    parser.add_argument('-rp','--root_path', help='root path', 
            default = "/data/home/zhouqiongyi/Algonauts2021_devkit/", type=str)
    parser.add_argument('-ddir','--data_dir', help='data path', 
            default = "/data/home/zhouqiongyi/dataset/univ_wenlan/", type=str)
    parser.add_argument('-rd','--result_dir', help='saves predicted fMRI activity',
            default = 'results_textual', type=str)
    parser.add_argument('-model','--model',help='model list', nargs='+')
    parser.add_argument('-layer','--layer',help='using features of which layer.', 
            default = 1, type=int)
    parser.add_argument('-m','--mode',help='hyper_tune, train, test', default = 'train', type=str)
    parser.add_argument('-cp', '--csv_path',help='name of csv file', 
            default = 'summary', type=str)
    parser.add_argument('-sid', '--sub_id',help='subject id: [P01,M02,M04,M07,M08,M09,M14,M15]', 
            default = 'M02', type=str)
    parser.add_argument('-exp', '--exp',help='experiment', 
            choices=['examples_180concepts_sentences', 'examples_384sentences'], type=str)
    # parser.add_argument('-ti_st', '--time_stamp', help='only need in mode test', 
    #         default=' ', type=str)
    parser.add_argument('-cvi','--cv_inner',help='cv of training set.', 
            default = 5, type=int)
    parser.add_argument('-cvo','--cv_outer',help='cv of the whole dataset.', 
            default = 5, type=int)
    args = vars(parser.parse_args())
    print(args)
    
    hyper_suffix = 'layer_'+str(args['layer'])
    # time_stamp = time.strftime("%m%d%H%M%S", time.localtime())
    mode = args['mode']
    model_list = [ele for ele in args['model'][0].split(',')]
    results_dir = os.path.join(args['root_path'], 
                            args['result_dir'], '_'.join(model_list), 
                            '_'.join([args['sub_id'], args['exp'].split('_',1)[-1]]))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    csv_save_path = os.path.join(results_dir, args['csv_path']+'.csv')
    data_to_csv = {'r2_fold_aver': 0}

    if mode == 'hyper_tune':
        # suffix = '_'.join([mode, time_stamp, hyper_suffix])
        suffix = '_'.join([mode, hyper_suffix])
        print('Load training data.')
        fea, fmri, dim_list = load_data(args['data_dir'], 
                                        model_list, args['layer'], 
                                        subject_id=args['sub_id'],
                                        exp=args['exp'])
        print('Training data loaded.')
        if args['layer'] >= 0:
            print('Using features of layer-%d'%args['layer'])
        else:
            print('Using features of all layers')
        print('feature and fmri shape', fea.shape, fmri.shape)
        
        kf = KFold(n_splits=args['cv_outer'], shuffle=True, random_state=1)
        # # # basic settings # # # # 
        alphas = np.logspace(1, 20, 20)
        solver = "random_search"
        n_iter = 20
        n_targets_batch = 200
        n_alphas_batch = 5
        n_targets_batch_refit = 200
        solver_params = dict(n_iter=n_iter, alphas=alphas,
                            n_targets_batch=n_targets_batch,
                            n_alphas_batch=n_alphas_batch,
                            n_targets_batch_refit=n_targets_batch_refit)

        r2_aver = np.zeros(fmri.shape[-1])
        r2_split_aver = np.zeros((2, fmri.shape[-1]))
        # # # # # # # # # # # # # # 
        for i, (train_idx, test_idx) in enumerate(kf.split(fea)):
            print('Cross validation, Fold {}'.format(i+1))
            fea_train = fea[train_idx]
            fea_test = fea[test_idx]
            fmri_train = fmri[train_idx]
            fmri_test = fmri[test_idx]
            print('fea_train and fmri_train shape', fea_train.shape, fmri_train.shape)
            print('fea_test and fmri_test shape', fea_test.shape, fmri_test.shape)
            # # # # normalize fmri # # # 
            scaler = StandardScaler()
            fmri_train = scaler.fit_transform(fmri_train)
            # zero-mean is necessary for accurate score split.
            fmri_test = scaler.fit_transform(fmri_test)
            print('y_test_mean=', fmri_test.mean())
            # # # # # # # # # # # # # # 
            backend = set_backend("torch_cuda", on_error="warn")
            print('Build regresser')

            mkr_model = MultipleKernelRidgeCV(kernels="precomputed", solver=solver,
                                            solver_params=solver_params, 
                                            cv=args['cv_inner'])

            preprocess_pipeline = make_pipeline(
                StandardScaler(with_mean=True, with_std=False),
                Kernelizer(kernel="linear"),
            )
            start_and_end = np.concatenate([[0], np.cumsum(dim_list)])
            slices = [
                slice(start, end)
                for start, end in zip(start_and_end[:-1], start_and_end[1:])
            ]
            kernelizers_tuples = [(name, preprocess_pipeline, slice_)
                                for name, slice_ in zip(model_list, slices)]
            column_kernelizer = ColumnKernelizer(kernelizers_tuples)
            pipeline = make_pipeline(
                column_kernelizer,
                mkr_model,
            )

            print('Regressor built.')
            start_time = time.time()
            pipeline.fit(fea_train, fmri_train)
            print('Model fitting spends %.4f Min'%((time.time()-start_time)/60))

            print('Save results to .pickle')
            model_dir = os.path.join(results_dir, 'model', suffix, str(i+1))
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            with open(os.path.join(model_dir, 'model.pickle'), 'wb') as f:
                pickle.dump(pipeline, f)
            print('Model saved.')

            print('Test start')
            scores_test = pipeline.score(fea_test, fmri_test)
            scores_test = backend.to_numpy(scores_test)
            n_voxels = fmri_train.shape[-1]
            print('Test finished')

            res_dir = os.path.join(results_dir, 'res', suffix, str(i+1))
            if not os.path.exists(res_dir):
                os.makedirs(res_dir)
            np.save(os.path.join(res_dir, 'r2.npy'), scores_test)
            print('Results saved.')
            r2_aver += scores_test

            # vis contributions of different features.
            Y_test_pred_split = pipeline.predict(fea_test, split=True)
            split_scores_test = r2_score_split(fmri_test, Y_test_pred_split)
            split_scores_test = backend.to_numpy(split_scores_test)
            np.save(os.path.join(res_dir, 'r2_split.npy'), 
                    split_scores_test)
            r2_split_aver += split_scores_test
        r2_aver /= args['cv_outer']
        r2_split_aver /= args['cv_outer']
        np.save(os.path.join(res_dir, 'r2_fold_aver.npy'), r2_aver)
        np.save(os.path.join(res_dir, 'r2_split_fold_aver.npy'), r2_split_aver)

    elif mode == 'test':
        # time_stamp = args['time_stamp']
        backend = set_backend("torch_cuda", on_error="warn")
        suffix = '_'.join(['hyper_tune', hyper_suffix])

        print('Load data.')
        fea, fmri, dim_list = load_data(args['data_dir'], 
                                        model_list, args['layer'], 
                                        subject_id=args['sub_id'],
                                        exp=args['exp'])
        r2_aver = np.zeros(fmri.shape[-1])
        r2_split_aver = np.zeros((2, fmri.shape[-1]))
        kf = KFold(n_splits=args['cv_outer'], shuffle=True, random_state=1)
        for i, (train_idx, test_idx) in enumerate(kf.split(fea)):
            print('Cross validation, Fold {}'.format(i+1))
            fea_test = fea[test_idx]
            fmri_test = fmri[test_idx]
            # # # # normalize fmri # # # 
            scaler = StandardScaler()
            fmri_test = scaler.fit_transform(fmri_test)
            # # # # # # # # # # # # # # 

            model_dir = glob.glob(os.path.join(results_dir, 'model', suffix, str(i+1)))[0]
            res_dir = glob.glob(os.path.join(results_dir, 'res', suffix, str(i+1)))[0]
            with open(os.path.join(model_dir, 'model.pickle'), 'rb') as f:
                pipeline = pickle.load(f)

            scores_test = pipeline.score(fea_test, fmri_test)
            scores_test = backend.to_numpy(scores_test)
            np.save(os.path.join(res_dir, 'r2.npy'), scores_test)
            r2_aver += scores_test

            Y_test_pred_split = pipeline.predict(fea_test, split=True)
            split_scores_test = r2_score_split(fmri_test, Y_test_pred_split)
            split_scores_test = backend.to_numpy(split_scores_test)
            np.save(os.path.join(res_dir, 'r2_split.npy'), split_scores_test)
            r2_split_aver += split_scores_test
        r2_aver /= args['cv_outer']
        r2_split_aver /= args['cv_outer']
        np.save(os.path.join(res_dir, 'r2_fold_aver.npy'), r2_aver)
        np.save(os.path.join(res_dir, 'r2_split_fold_aver.npy'), r2_split_aver)

    print('plot the hist.')
    plt.figure()
    plt.hist(r2_aver, weights=(np.ones_like(r2_aver)/float(r2_aver.shape[0])), bins=50)
    plt.yscale('log')
    plt.savefig(os.path.join(res_dir, 'hist_r2_fold_aver.png'))
    plt.close()

    print('write csv')
    data_to_csv['r2_fold_aver'] = r2_aver.mean()
    data_to_csv['mode'] = mode
    data_to_csv['suffix'] = suffix
    write_csv(data_to_csv, 
        ['mode', 'suffix', 'r2_fold_aver'], 
        csv_save_path)
    print('Completed.')

if __name__ == "__main__":
    main()
