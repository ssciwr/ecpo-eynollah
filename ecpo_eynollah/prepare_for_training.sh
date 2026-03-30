#!/usr/bin/env bash

# script to prepare data for training Eynollah
# before running this script, make sure that you have:
# 1. The JSON-MIN file downloaded from LabelStudio
# 2. The folder where you want to put all data in

LS_JSON_FILE="$1"
ROOT_DIR="$2"
ADS_SEPARATION="$3"
WITH_HEADINGS="$4"
BUFFER_SIZE="$5"
TRAINING_RATIO="$6"

if [ -z "$LS_JSON_FILE" ] || [ -z "$ROOT_DIR" ]; then
  echo "Usage: $0 <path_to_labelstudio_json> <root_directory> <ads_separation> <with_headings> <buffer_size_optional> <training_ratio_optional>"
  exit 1
fi

if [ -z "$ADS_SEPARATION" ]; then
  ADS_SEPARATION=false
fi

if [ -z "$WITH_HEADINGS" ]; then
  WITH_HEADINGS=true
fi

if [ -z "$BUFFER_SIZE" ]; then
  BUFFER_SIZE=10
fi

if [ -z "$TRAINING_RATIO" ]; then
  TRAINING_RATIO=0.8
fi

# create necessary directories
mkdir -p "$ROOT_DIR/jb-org"
mkdir -p "$ROOT_DIR/data"
mkdir -p "$ROOT_DIR/labelstudio/no-color"
mkdir -p "$ROOT_DIR/out"


# download original images, using LabelStudio JSON-MIN file
download-ls-imgs \
    --json-file "$LS_JSON_FILE" \
    --output-dir "$ROOT_DIR/jb-org" \
    --overwrite

# generate labeled data, using LabelStudio JSON-MIN file
if [ "$WITH_HEADINGS" = true ]; then
  HEADING_FLAG="--withheadings"
else
  HEADING_FLAG="--no-withheadings"
fi

labelstudio2png \
    --input "$LS_JSON_FILE" \
    --output "$ROOT_DIR/labelstudio/no-color" \
    --no-color \
    --overwrite \
    --buffer-size "$BUFFER_SIZE" \
    $HEADING_FLAG

# split data into training and evaluation sets
if [ "$ADS_SEPARATION" = true ]; then
  ADS_FLAG="--ads-separation"
else
  ADS_FLAG="--no-ads-separation"
fi

prepare-data \
    --img-dir "$ROOT_DIR/jb-org" \
    --labeled-dir "$ROOT_DIR/labelstudio/no-color" \
    --train-ratio "$TRAINING_RATIO" \
    $ADS_FLAG \
    --out-dir "$ROOT_DIR/data"

# prepare output folders in case of ads separation
if [ "$ADS_SEPARATION" = true ]; then
  mkdir -p "$ROOT_DIR/out/ads-heavy"
  mkdir -p "$ROOT_DIR/out/text-heavy"
else
  mkdir -p "$ROOT_DIR/out/mixed"
fi

echo "Data preparation complete. Data is in $ROOT_DIR/data"