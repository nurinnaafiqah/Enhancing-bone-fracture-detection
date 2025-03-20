# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 00:11:18 2025

@author: nurin afiqah
"""

import os
import shutil

# Paths
images_folder = r"C:\Users\afiqa\OneDrive\Documents\Nurin's\FINAL YEAR PROJECT\FYP DATASETS\valid\canny edge"
labels_folder = r"C:\Users\afiqa\OneDrive\Documents\Nurin's\FINAL YEAR PROJECT\FYP DATASETS\valid\labels"
positive_folder = r"C:\Users\afiqa\OneDrive\Documents\Nurin's\FINAL YEAR PROJECT\FYP DATASETS\structured canny edge\valid\positive"
negative_folder = r"C:\Users\afiqa\OneDrive\Documents\Nurin's\FINAL YEAR PROJECT\FYP DATASETS\structured canny edge\valid\negative"

# Create output folders if they don't exist
os.makedirs(positive_folder, exist_ok=True)
os.makedirs(negative_folder, exist_ok=True)

# Process each file
for label_file in os.listdir(labels_folder):
    if label_file.endswith(".txt"):
        label_path = os.path.join(labels_folder, label_file)
        image_name = os.path.splitext(label_file)[0] + ".jpg"  # Assuming image format is .jpg
        image_path = os.path.join(images_folder, image_name)

        # Check if corresponding image exists
        if not os.path.exists(image_path):
            print(f"Image {image_name} not found for label {label_file}")
            continue

        # Check label file content
        with open(label_path, "r") as f:
            content = f.read().strip()

        if content:  # If label file has content, it's positive
            shutil.copy(image_path, os.path.join(positive_folder, image_name))
        else:  # If label file is empty, it's negative
            shutil.copy(image_path, os.path.join(negative_folder, image_name))

print("Images successfully organized into positive and negative folders.")
