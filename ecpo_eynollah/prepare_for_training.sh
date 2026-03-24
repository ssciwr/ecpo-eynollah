#!/usr/bin/env bash

# script to prepare data for training Eynollah
# before running this script, make sure that you have:
# 1. The JSON-MIN file downloaded from LabelStudio
# 2. The folder where you want to put all data in

LS_JSON_FILE="$1"
ROOT_DIR="$2"
BUFFER_SIZE="$3"
TRAINING_RATIO="$4"

if [ -z "$LS_JSON_FILE" ] || [ -z "$ROOT_DIR" ]; then
  echo "Usage: $0 <path_to_labelstudio_json> <root_directory> <buffer_size> <training_ratio_optional>"
  exit 1
fi

if [ -z "$BUFFER_SIZE" ]; then
  BUFFER_SIZE=10
fi

if [ -z "$TRAINING_RATIO" ]; then
  TRAINING_RATIO=0.8
fi

# create necessary directories
mkdir -p "$ROOT_DIR/jb-org"
mkdir -p "$ROOT_DIR/train"
mkdir -p "$ROOT_DIR/eval"
mkdir -p "$ROOT_DIR/labelstudio/no-color"
mkdir -p "$ROOT_DIR/out"


# download original images, using LabelStudio JSON-MIN file
download-ls-imgs \
    --json-file "$LS_JSON_FILE" \
    --output-dir "$ROOT_DIR/jb-org" \
    --overwrite

# generate labeled data, using LabelStudio JSON-MIN file
labelstudio2png \
    --input "$LS_JSON_FILE" \
    --output "$ROOT_DIR/labelstudio/no-color" \
    --no-color \
    --overwrite \
    --buffer-size "$BUFFER_SIZE"

# split data into training and evaluation sets
prepare-data \
    --img-dir "$ROOT_DIR/jb-org" \
    --labeled-dir "$ROOT_DIR/labelstudio/no-color" \
    --train-ratio "$TRAINING_RATIO" \
    --out-train-dir "$ROOT_DIR/train" \
    --out-eval-dir "$ROOT_DIR/eval" \

echo "Data preparation complete. Training data is in $ROOT_DIR/train and evaluation data is in $ROOT_DIR/eval"