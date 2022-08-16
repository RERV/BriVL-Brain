# Run description of the validation experiments on the text dataset.


**1. data** The features of the textual data need to be stored in ``encoding_wenlan/text_feature/`` for the 8 subjects (codenamed [P01,M02,M04,M07,M08,M09,M14,M15]) with neural The responses have been uploaded to ``/home/haoyu_lu/NATURE2022/encoding_wenlan/datasets/textual_dataset``.

**2. results** The results will be saved in ``encoding_wenlan/results_textual/wenlan_bert/subjectid_expid``, where subjectid is a different subject designator and expid is different experiment code (180concepts_sentences or 384sentences both, the former is words for stimuli, the latter is sentences for stimuli). results for M02 and M04 have been uploaded (model parameters are too large, not uploaded, only predicted results are uploaded for results visualization).

**3. run**
* Before running ``perform_encoding_textual_banded.py`` line 15, the himalaya library import method may need to be adjusted here.
* ```perform_encoding_textual_banded.py`` is the prediction model, you can specify subjects, experiment type, and feature layer. ```plot_layer_textual.py`` is responsible for plotting **per subject** a line graph of prediction accuracy vs. number of feature layers for each experiment type, saved to ``/home/haoyu_lu/NATURE2022/encoding_wenlan/results_textual/ wenlan_bert/subjectid_expid/```. ``plot_layer_textual_sub_aver.py`` is responsible for plotting a line graph of the average** prediction accuracy vs. number of feature layers on all subjects for each experiment type**, saved to ``encoding_wenlan/results_ textual/wenlan_bert/``. Note: ``perform_encoding_textual_banded.py`` makes predictions for all voxels, and when visualizing, selects different voxel filters based on args.sel into whole-brain voxels (all), r2>0 voxels (pos), and r2>0 and non-cerebellar voxels (90pos).
* Switch to the path ``encoding_wenlan/textual_code`` and run ``shell_textual.sh``. The script includes the procedure from model prediction to result visualization on the remaining 6 subjects.
