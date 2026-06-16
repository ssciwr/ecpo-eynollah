#!/usr/bin/env bash

# Script to infer Eynollah model on unseen data
# has to run in the environment where Eynollah is installed

MODEL="$1"
IMG_DIR="$2"
OUTPUT_DIR="$3"

if [ -z "$MODEL" ] || [ -z "$IMG_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: $0 <model_path> <image_directory> <output_directory>"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

for img in "$IMG_DIR"/*.png; do
  filename=$(basename "$img")
  output_file="$OUTPUT_DIR/${filename%.png}_inferred.png"
  
  echo "Processing $filename..."
  
  eynollah-training inference \
    -m "$MODEL" \
    -i "$img" \
    -p \
    -s "$output_file"
  
  echo "Saved inference results to $output_file"
done