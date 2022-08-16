# split scores across all layers for each subject and each experiment
import os
import glob
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-sid', '--sub_id',help='subject id', 
        default = 'M02', type=str)
parser.add_argument('-exp', '--exp',help='experiment', 
        choices=['180concepts_sentences', '384sentences'], type=str)
parser.add_argument('-sel', '--selection', help='voxel selection',
        type=str)
parser.add_argument('-rp', '--root_path', default='/data/home/zhouqiongyi/Algonauts2021_devkit',
        type=str)
parser.add_argument('-ddir','--data_dir', default='/data/home/zhouqiongyi/dataset/univ_wenlan',
        type=str)
args = parser.parse_args()

from matplotlib import pyplot
plt.style.use('seaborn-poster')
palette = pyplot.get_cmap('Set2')

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 23,
}

fig=plt.figure(figsize=(16, 10))


root_path = os.path.join(args.root_path, 'results_textual/wenlan_1400w_coco_bert/%s_%s'%(args.sub_id, args.exp))
data_dir = os.path.join(args.data_dir, '%s/examples_%s.mat'%(args.sub_id, args.exp))

meta = sio.loadmat(data_dir)['meta'][0, 0]
roi_label = meta['roiMultimaskAAL'].squeeze()   #(voxel_num)

r2_list = []
r2_split_wenlan_list = []
r2_split_bert_list = []
mean_res = sio.loadmat(data_dir)['examples'].mean(0)
# exclude voxels whose average response to stimuli is zero.
nonzero_vox = np.where(mean_res!=0)[0]

for layer in range(1,13,1):
    res_folder = glob.glob(os.path.join(root_path, 
                        'res/hyper_tune_layer_%d/'%layer))

#     print(res_folder)
#     print(os.path.join(root_path, 
                        # 'res/hyper_tune_layer_%d/'%layer))
    res_folder = res_folder[-1]
    print(res_folder)
    kfold_res_path = os.path.join(res_folder, '5')
    r2_fold_aver = np.load(os.path.join(kfold_res_path, 'r2_fold_aver.npy'))
    r2_split_fold_aver = np.load(os.path.join(kfold_res_path, 'r2_split_fold_aver.npy'))

    if args.selection == 'all':
        voxel_mask = nonzero_vox
    elif args.selection == 'pos':
        # exclude the voxels whose r2 <= 0 (worse prediction)
        voxel_mask = np.intersect1d(np.where(r2_fold_aver>0)[0], nonzero_vox)
    elif args.selection == '90pos':
        # exclude cerebellum voxels and the voxels whose r2 <= 0 (worse prediction) 
        voxel_mask = np.intersect1d(np.where(r2_fold_aver>0)[0], nonzero_vox)
        roi_mask = np.intersect1d(np.where(roi_label<=90)[0], np.where(roi_label>0)[0])
        voxel_mask = np.intersect1d(voxel_mask, roi_mask)
    r2_split_mask = r2_split_fold_aver[:, voxel_mask]
    r2_mask = r2_fold_aver[voxel_mask]

    r2_list.append(r2_mask.mean())
    r2_split_wenlan_list.append(r2_split_mask[0].mean())
    r2_split_bert_list.append(r2_split_mask[1].mean())


x = np.arange(1, len(r2_split_wenlan_list)+1)
plt.plot(x, r2_split_wenlan_list, label='BriVL', color='royalblue')
plt.plot(x, r2_split_bert_list, label='Bert', color='darkorange')
plt.plot(x, r2_list, label='Both', color='darkgreen', linestyle='--')
plt.xlabel('Layer')
plt.ylabel('Mean $R^2$')
plt.legend()
ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(1))
plt.show()
plt.savefig(os.path.join(root_path, 'avgr2_layer_%s.png'%args.selection), bbox_inches='tight')
plt.close()