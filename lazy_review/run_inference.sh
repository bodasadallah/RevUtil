#!/bin/bash

# Lazy Review Inference Script
# This script runs inference on the lazy thinking classification task

# Configuration
MODEL_PATH="sukannya/lazy_review_plus_emnlp_Qwen2.5-7B-Instruct"
DATA_FILE="data/lazy_thinking_fine_grained_test_with_eg.jsonl"
OUTPUT_DIR="outputs"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "Running Lazy Review Inference"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATA_FILE"
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="

# Run inference
python eval.py \
    --dataset $DATA_FILE \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR

echo "=========================================="
echo "Inference completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
