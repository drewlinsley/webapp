cd ../linear_models

TARGET="TARDBP"
GPU=6
VERSION=13
PROBE="InChI=1S/C8H5F3N2OS/c9-8(10,11)14-4-1-2-5-6(3-4)15-7(12)13-5/h1-3H,(H2,12,13)"
CUDA_VISIBLE_DEVICES=$GPU python no_reduce_no_norm.py $TARGET $VERSION

cd ../CLIP_prefix_caption

python prepare_data.py
accelerate launch train.py --only_prefix --train_data=galactica_normalized_preproc_for_train.npz --eval_data=galactica_normalized_preproc_for_test.npz --log
