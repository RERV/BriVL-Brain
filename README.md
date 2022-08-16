# BriVL-Brain
## The code repository for our paper "Multimodal foundation models are better simulators of human brain than unimodal ones"

#  Multimodal foundation models are better simulators of the human brain
## 0.Framework
```
encoding_wenlan:.
│  encoding_wenlan.py
|  README.md
│
├─feature_extraction
│  ├─alexnet.py
│  ├─generate_features.py
│  └─utils.py
|
├─datasets
│  ├─responses
│  |   └─S01_responses.hdf
│  └─stimuli
|      ├─train_xx.hdf
|      └─test.hdf
|
├─feature
|
└─results
```

## 1. Introduction to the data
Stimuli videos are stored in . /datasets/stimuli/. The training set stimuli consist of 12 video stimuli, each train_xx.hdf has 9000 frames stored inside, size (512, 512, 3) images. The subjects watched 30 frames of images in one TR time, so the features of every 30 frames need to be averaged for feature extraction. A train_xx.hdf corresponds to 9000/30=300 samples, and 12 videos are 300*12=3600 samples. The neural response data is stored in . /datasets/responses/, and the size of the response data corresponding to the training set stimuli is (3600, 84038), indicating the number of samples and voxels, respectively. The size of the response data corresponding to the test stimuli is (10, 270, 84038), indicating the number of repetitions, the number of samples and the number of voxels, respectively, and we averaged the test responses over different repetitions as Ground-Truth during the fitting.

In the fitting, we also take into account the hemodynamic delay. That is, the neural response of the 2nd TR may contain both the stimulus information of the 1st TR and the 2nd TR, so the features are delayed in encoding_wenlan.py (see the function delay_transform for details).

## 2. Code run
### 2.1 Extracting network features

Take alexnet as an example. First run generate_features.py under feature_extraction. note: the convolutional layer feature map is too high dimensional, use pca to reduce the dimension and keep 99.9% of the principal components.
```shell
python3.7 . /feature_extraction/generate_features.py -fname='alexnet' -dim_rd='pca' -pca_ratio=0.999
```
The extracted features are stored in . /feature/alexnet/.

The Vineland model extracts the network features simply in . /feature/utils.py with the functions load_model, image_transform and feature_extraction.

### 2.2 Running the encoding model
```shell
python3.7 encoding_wenlan.py -dim_rd='pca' -m='hyper_tune' -model='wenlan' -layer=-1 -delays=0,2,3,4
```
The coding results (model, results, visualization) are saved in . /results/.


## 3.Requirments
```
python3.7
torch == 1.7.0
himalaya == 0.3.5
timm == 0.5.4
sklearn == 0.23.2
```
