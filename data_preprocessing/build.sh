#!/bin/bash

set -e # Stop at first error

# Directories
RAW_DATA_ROOT="data/TKR"
PROCESSED_DIR="data/TKR/processed"

# Raw / processed metadata
RAW_METADATA="TKR傷口分析0325.xlsx"
MASTER_METADATA_FILE="master_metadata.csv"

# Raw / processed images
RAW_IMG_DIRS="工作表1 07 08 09 10 11"
IMG_MAP_FILE="image_id_map.json"
PROCESSED_IMG_DIR="images"

# Train-test split
SPLIT_VERSION="original_v0"

echo "------------------------------------------------------"
echo "[1/2] Run builder: convert & check images and metadata"
echo "------------------------------------------------------"
python -m data_preprocessing.data_engineering.builder \
       --raw_data_root             "$RAW_DATA_ROOT" \
       --raw_metadata_file         "$RAW_METADATA" \
       --raw_img_dir_names          $RAW_IMG_DIRS \
       --output_root               "$PROCESSED_DIR" \
       --output_image_dirname      "$PROCESSED_IMG_DIR" \
       --output_metadata_filename  "$MASTER_METADATA_FILE" \
       --img_map_filename          "$IMG_MAP_FILE"

echo "----------------------------------------------------------------"
echo "[2/2] Run splitter: train-test split (version: "$SPLIT_VERSION")"
echo "----------------------------------------------------------------"
python -m data_preprocessing.data_engineering.splitter \
       --processed_dir             "$PROCESSED_DIR" \
       --img_map_filename          "$IMG_MAP_FILE" \
       --metadata_filename         "$MASTER_METADATA_FILE" \
       --split_version             "$SPLIT_VERSION"