#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an InceptionV3 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_inceptionv3_on_flowers.sh


# Where the dataset is saved to.
DATASET_DIR=/home/brucelau/workbench/data/leaf_photos

# Download the pre-trained checkpoint.
if [ ! -d "$DATASET_DIR" ]; then
  mkdir ${DATASET_DIR}
fi

# covert the dataset
python convert_custom_data.py \
  --dataset_name=leaves \
  --dataset_dir=${DATASET_DIR}
