#!/bin/bash

# Define the list of augmentation keys
augmentation_keys=(
    "RandomHorizontalFlip"
    "RandomVerticalFlip"
    "RandomRotation"
    "ColorJitter"
    "RandomAffine"
    "RandomPerspective"
    "RandomErasing"
    "RandomGrayscale"
    "RandomAffineWithResize"
    "RandomPosterize"
    "RandomSolarize"
    "RandomEqualize"
)

# Iterate through the list of augmentation keys and call the function
for index in "${!augmentation_keys[@]}"; do
    if [ "$index" -eq 6 ]; then
        transform_name="${augmentation_keys[index]}"
        CUDA_VISIBLE_DEVICES=1 python solutions/v6/solution.py --augmentation "$transform_name" --index "#$((index + 1))"
    fi
done
