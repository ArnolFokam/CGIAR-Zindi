#!/bin/bash

# Define your dictionary of models
declare -A models=(
    ["xcit_nano_12_p16_224"]="xcit_nano_12_p16_224"
    ["xcit_nano_12_p8_224"]="xcit_nano_12_p8_224"
    ["xcit_tiny_12_p16_224"]="xcit_tiny_12_p16_224"
    ["xcit_tiny_12_p8_224"]="xcit_tiny_12_p8_224"
    ["xcit_tiny_24_p8_224"]="xcit_tiny_24_p8_224"
    ["xcit_tiny_24_p16_224"]="xcit_tiny_24_p16_224"
    ["xcit_small_12_p8_224"]="xcit_small_12_p8_224"
    ["xcit_small_12_p16_224"]="xcit_small_12_p16_224"
    ["xcit_small_24_p8_224"]="xcit_small_24_p8_224"
    ["xcit_small_24_p16_224"]="xcit_small_24_p16_224"
)

# Initialize an incrementing index
index=1

# Loop through the keys and call the Python program
for key in "${!models[@]}"; do
    model_name="${models[$key]}"
    CUDA_VISIBLE_DEVICES=1 python solutions/v8/solution.py --index "#$index" --model_index "$index" --model_name "$model_name"
    ((index++))
done
