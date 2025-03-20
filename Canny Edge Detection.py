# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 00:08:35 2025

@author: afiqa
"""

import cv2
import os
from glob import glob

# Define paths
images_folder = r"C:\Users\afiqa\OneDrive\Documents\Nurin's\FINAL YEAR PROJECT\FYP DATASETS\valid\images"
output_folder = r"C:\Users\afiqa\OneDrive\Documents\Nurin's\FINAL YEAR PROJECT\FYP DATASETS\valid\canny edge"

# Make sure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Define Canny edge detection parameters
threshold1 = 100  # Lower threshold for edge detection
threshold2 = 200  # Upper threshold for edge detection

# Define new dimensions for resizing (e.g., 256x256)
new_width = 256
new_height = 256

# Process each image
for image_path in glob(os.path.join(images_folder, '*.jpg')):  # Adjust extension if needed
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Use grayscale for edge detection
    
    # Apply Canny edge detection
    edges = cv2.Canny(image, threshold1, threshold2)
    
    # Resize the image
    resized_edges = cv2.resize(edges, (new_width, new_height))
    
    # Save the processed image to the output folder
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, resized_edges)

print("Canny edge detection and resizing applied to all images.")
