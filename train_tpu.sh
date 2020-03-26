#!/bin/bash
set -e

export NOISY=1
export DEBUG=1
export G_LR=0.003
export D_LR=0.003

config="config-f"

data_dir= # gs://your_bucket/your_dataset_dir/your_dataset_name
dataset= # your_dataset_name
mirror=false
metrics=none

export TPU_HOST= # your host name
export TPU_NAME= # your tpu name 
cores= # your tpu cores
export MODEL_DIR= # gs://your_bucket/your_results_dir/your_dataset_name-resolution_size
export BATCH_PER= # gpu-wise batch size (batch ubset per gpu) 
export BATCH_SIZE=$(($BATCH_PER * $cores))
export RESOLUTION= # resolution_size

set -x
exec python3 -m pdb -c continue run_training.py --num-gpus="${cores}" --data-dir="${data_dir}" --config="${config}" --dataset="${dataset}" --mirror-augment="${mirror}" --metrics="${metrics}" "$@"
