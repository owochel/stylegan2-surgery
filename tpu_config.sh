#!/bin/bash
set -e


#
#
#   ~~ aydao tpu config ~~
#   ~~~   aydao 2020   ~~~
#
#   Massive credit to shawwn, gwern, arfa, l4rz
#   tensorfork more broadly, and the "tpu podcast" discord
#
#
#
# This is the mega-long run trained on danbooru2019-s (512px)
#   basis was shawwn's stable tpu branch of the stylegan2 fork
#   https://github.com/shawwn/stylegan2/tree/working
#
# MAJOR CHANGES TO ARCHITECTURE / TRAINING:
#   set G loss to G_logistic_ns (that is, no pathreg)
#   turn off stylemixing, in G_main 
#       style_mixing_prob       = None,
#   increase BOTH latentsize and dlatent size to 1024 
#   normalize_latents set to False
#   increase fmaps in G:
#       fmap_base           = 32 << 10
#       fmap_max            1024
#   mbstd_group_size    set to 32
#   mbstd_num_features  set to 4
#   set gamma to 5.0 (used to be  --gamma=10.0, lowered at ~4 million)
#   learning rate lowered late in training (see below)
#   topk added at ~2.3 million
#   switched from RANDOM_CROP to TOP_CROP at 
#   no noconst (just normal const ... lol I thought it was noconst, oh well)
#   did not use mirror augmentation
#   used figures, portraits, and hand data augmentation (PALM and the rest) 
#       https://www.gwern.net/Crops
#   (data augs were added fairly early in training)
#
#

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/tfk/lib}"

export NOISY=1
export DEBUG=1

config="config-f" # StyleGAN 2

export IMAGENET_UNCONDITIONAL=1
export LABEL_SIZE=0
export LABEL_BIAS=0
export CROP_PADDING=0
# export RANDOM_CROP=1 # Note: this was enabled for the first ~4.5 million 
export TOP_CROP=1 # Note: this was enabled starting at 4.5 million
export G_LR=0.0004 # Note: started at export G_LR=0.002 for first ~3 million
export D_LR=0.0003 # Note: started at export D_LR=0.002 for first ~3 million
export G_LR_MULT=1.0
export D_LR_MULT=1.0

data_dir=your gs:// data bucket here # placeholder -- refer to shawwn's tpu code
dataset=animefaces # placeholder -- refer to shawwn's tpu code
mirror=false # Note: this was false for the 
metrics=none

export TPU_SPLIT_COMPILE_AND_EXECUTE=1
export TF_TPU_WATCHDOG_TIMEOUT=1800
export TPU_HOST=10.255.128.3
export TPU_NAME=your tpu name / IP here # placeholder -- refer to shawwn's tpu code
cores=32

export IMAGENET_TFRECORD_DATASET='your gs:// data bucket here (tfrecords)'
export MODEL_DIR='your gs:// model directory bucket here (checkpoint destination)'
export BATCH_PER=4
export BATCH_SIZE=$(($BATCH_PER * $cores))
export RESOLUTION=512

set -x
python3 run_training.py --num-gpus="${cores}" --data-dir "${data_dir}" --config="${config}" --dataset="${dataset}" --gamma=5.0 --mirror-augment="${mirror}" --metrics="${metrics}" "$@"