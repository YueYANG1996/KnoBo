#!/bin/bash

# This is a bash script that extracts features from the datasets listed below.
DATASET_DIR=$1
IMAGE_DIR=$2

XRAY_DATASET_LIST=('NIH-sex' 'NIH-age' 'NIH-pos' 'CheXpert-race' 'NIH-CheXpert' 'pneumonia' 'COVID-QU' 'NIH-CXR' 'open-i' 'vindr-cxr')
SKIN_LESION_DATASET_LIST=('ISIC-sex' 'ISIC-age' 'ISIC-site' 'ISIC-color' 'ISIC-hospital' 'HAM10000' 'BCN20000' 'PAD-UFES-20' 'Melanoma' 'UWaterloo')

# for loop to run the python script for each dataset in the XRAY_DATASET_LIST
for dataset in "${XRAY_DATASET_LIST[@]}"; do
    python modules/extract_features.py --model_name whyxrayclip --dataset_dir "$DATASET_DIR" --image_dir "$IMAGE_DIR" --dataset_name "$dataset"
done

# for loop to run the python script for each dataset in the SKIN_LESION_DATASET_LIST
for dataset in "${SKIN_LESION_DATASET_LIST[@]}"; do
    python modules/extract_features.py --model_name whylesionclip --dataset_dir "$DATASET_DIR" --image_dir "$IMAGE_DIR" --dataset_name "$dataset"
done