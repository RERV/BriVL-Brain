# split scores across all layers averaged on all subjects
import os
import glob
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-exp', '--exp',help='experiment', 
        choices=['180concepts_sentences', '384sentences'], type=str)
parser.add_argument('-sel', '--selection', help='voxel selection',
        type=str)
parser.add_argument('-rp', '--root_path', default='/home/haoyu_lu/NATURE2022/encoding_wenlan/',
        type=str)
parser.add_argument('-ddir', '--data_dir', default='/home/haoyu_lu/NATURE2022/encoding_wenlan/datasets/textual_dataset',
        type=str)
args = parser.parse_args()

r2_sub_aver = []
r2_split_wenlan_sub_aver = []
r2_split_bert_sub_aver = []

for sub_id in ['P01','M02','M04','M07','M08','M09','M14','M15']:
# for sub_id in ['P01','M02','M04','M07','M08','M14','M15']:
# for sub_id in ['P01','M02','M04', 'M09','M14','M15']:
    root_path = os.path.join(args.root_path, 'results_textual/wenlan_1400w_coco_bert/%s_%s'%(sub_id, args.exp))
    data_dir = os.path.join(args.data_dir, '%s/examples_%s.mat'%(sub_id, args.exp))

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


    r2_sub_aver.append(r2_list)
    r2_split_wenlan_sub_aver.append(r2_split_wenlan_list)
    r2_split_bert_sub_aver.append(r2_split_bert_list)

r2_sub_aver = np.array(r2_sub_aver)
r2_split_wenlan_sub_aver = np.array(r2_split_wenlan_sub_aver)
r2_split_bert_sub_aver = np.array(r2_split_bert_sub_aver)

np.save('r2_sub_aver_layer', r2_sub_aver)
np.save('r2_split_wenlan_sub_aver_layer', r2_split_wenlan_sub_aver)
np.save('r2_split_bert_sub_aver_layer', r2_split_bert_sub_aver)


# print('r2_split_bert_sub_aver shape {}'.format(r2_split_bert_sub_aver.shape))
# from scipy import stats
# stats.ttest_rel(r2_split_wenlan_sub_aver[:, 11], r2_split_bert_sub_aver[:, 11])
# exit()
x = np.arange(1, len(r2_split_wenlan_list)+1)
plt.errorbar(x, r2_split_wenlan_sub_aver.mean(0), yerr=r2_split_wenlan_sub_aver.std(0), 
        label='BriVL', color='royalblue', lw = 2, elinewidth=2, ms=7, capsize=3)
plt.errorbar(x, r2_split_bert_sub_aver.mean(0), yerr=r2_split_bert_sub_aver.std(0),
        label='Bert', color='darkorange', lw = 2, elinewidth=2, ms=7, capsize=3)
plt.errorbar(x, r2_sub_aver.mean(0), yerr=r2_sub_aver.std(0), 
        label='Both', color='darkgreen', linestyle='--', lw = 2, elinewidth=2, ms=7, capsize=3)


plt.xlabel('Layer')
plt.ylabel('Mean $R^2$')
plt.legend()
ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(1))
plt.show()
plt.savefig(os.path.join(args.root_path, 
            'results_textual/wenlan_bert/roi_avgr2_layer_%s_%s.png'%(args.exp, args.selection)), 
            bbox_inches='tight')
plt.close()

exit()

from matplotlib import pyplot
plt.style.use('seaborn-poster')
palette = pyplot.get_cmap('Set2')
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 23,
}

fig=plt.figure(figsize=(16, 10))
iters=list(range(1, 13))


color=palette(2)
ax=fig.add_subplot(1,1,1)   
avg=np.mean(r2_split_wenlan_sub_aver,axis=0)
std=np.std(r2_split_wenlan_sub_aver,axis=0) #/ 5
r1 = list(map(lambda x: x[0]-x[1], zip(avg, std)))
r2 = list(map(lambda x: x[0]+x[1], zip(avg, std)))
ax.plot(iters, avg, color=color,label="Wenlan",linewidth=3.0)
ax.fill_between(iters, r1, r2, color=color, alpha=0.2)

color=palette(1)
avg=np.mean(r2_split_bert_sub_aver,axis=0)
std=np.std(r2_split_bert_sub_aver,axis=0) #/ 5
r1 = list(map(lambda x: x[0]-x[1], zip(avg, std)))
r2 = list(map(lambda x: x[0]+x[1], zip(avg, std)))
ax.plot(iters, avg, color=color,label="Bert",linewidth=3.0)
ax.fill_between(iters, r1, r2, color=color, alpha=0.2)

color=palette(0)
avg=np.mean(r2_sub_aver,axis=0)
std=np.std(r2_sub_aver,axis=0) #/ 5
r1 = list(map(lambda x: x[0]-x[1], zip(avg, std)))
r2 = list(map(lambda x: x[0]+x[1], zip(avg, std)))
ax.plot(iters, avg, color=color,label="Both",linewidth=3.0)
ax.fill_between(iters, r1, r2, color=color, alpha=0.2)

ax.legend(loc='upper left',prop=font1)
# plt.legend()
plt.tick_params(labelsize=18)
ax.set_xlabel('Layer',fontsize=22)
ax.set_ylabel('Mean $R^2$',fontsize=22)


plt.savefig(os.path.join(args.root_path, 
            'results_textual/wenlan_bert/new_2_avgr2_layer_%s_%s.png'%(args.exp, args.selection)), 
            bbox_inches='tight')
