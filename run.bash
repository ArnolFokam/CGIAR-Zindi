#!/bin/bash

# Define your list of transformations
transformations=(
    "log" 
    "sqrt" 
    "inv" 
    "exp" 
    "rank" 
    "pow"
)

# Set the model name
model_name="xcit_nano_12_p16_224"

# Set the label transformation
label_transform="log"  # Change this to the desired label transformation

# Initialize an incrementing index
index=1

python setup.py install

# Loop through the list and call the Python program
for transformation in "${transformations[@]}"; do
    CUDA_VISIBLE_DEVICES=0 echo python solutions/v12/solution.py --index "#$index" --model_name "$model_name" --label_transform "$transformation"
    ((index++))
done
