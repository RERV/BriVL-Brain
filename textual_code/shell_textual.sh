# for sid in 'M07' 'M08'  'M02' 'M04'  'P01' 'M09' 'M14' 'M15'
for sid in 'M15'
do
# for exp in 'examples_180concepts_sentences' 'examples_384sentences'
for exp in  'examples_384sentences'
do
for ((layer=1;layer<13;layer++))
do
CUDA_VISIBLE_DEVICES=3 python perform_encoding_textual_banded.py \
-model='wenlan_1400w_flickr','bert' \
-layer=$layer \
-m='hyper_tune' \
-sid=$sid \
-exp=$exp \
-cvi=5 \
-cvo=5 \
-rp='/home/haoyu_lu/NATURE2022/encoding_wenlan/' \
-ddir='/home/haoyu_lu/NATURE2022/encoding_wenlan/datasets/textual_dataset'
done
done
done

# for sid in   'M07' 'M08'  'M02' 'M04'  'P01' 'M09' 'M14' 'M15'
# do
# # for exp in '180concepts_sentences' '384sentences'
# for exp in '384sentences'
# do
# for sel in 'pos' 'all' '90pos'
# do
# python plot_layer_textual.py -sid=$sid \
# -exp=$exp -sel=$sel \
# -rp='/home/haoyu_lu/NATURE2022/encoding_wenlan/' \
# -ddir='/home/haoyu_lu/NATURE2022/encoding_wenlan/datasets/textual_dataset/'
# done
# done
# done

# for exp in '180concepts_sentences' '384sentences'
# for exp in '384sentences'
# do
# # for sel in 'pos' 'all' '90pos'
# for sel in '90pos'
# do
# python plot_layer_textual_sub_aver.py \
# -exp=$exp -sel=$sel \
# -rp='/home/haoyu_lu/NATURE2022/encoding_wenlan/' \
# -ddir='/home/haoyu_lu/NATURE2022/encoding_wenlan/datasets/textual_dataset'
# done
# done