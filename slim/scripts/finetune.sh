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

# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/home/brucelau/workbench/checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/home/brucelau/workbench/leafinception

# Where the dataset (TFRecords format)is saved to.
DATASET_DIR=/home/brucelau/workbench/dataset/leaf_photos

  # Fine-tune all the new layers for 500 steps.
  python train_image_classifier.py \
    --train_dir=${TRAIN_DIR}/all \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=inception_v3 \
    --checkpoint_path=${TRAIN_DIR} \
    --max_number_of_steps=1000 \
    --batch_size=32 \
    --learning_rate=0.001 \
    --learning_rate_decay_type=exponential \
    --save_interval_secs=60 \
    --save_summaries_secs=60 \
    --log_every_n_steps=10 \
    --optimizer=rmsprop \
    --weight_decay=0.0004

  # Run evaluation.
  python eval_image_classifier.py \
    --checkpoint_path=${TRAIN_DIR}/all \
    --eval_dir=${TRAIN_DIR}/all \
    --dataset_name=flowers \
    --dataset_split_name=validation \
    --dataset_dir=${DATASET_DIR} \
    --model_name=inception_v3
