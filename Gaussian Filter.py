# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 23:48:31 2025

@author: nurin afiqah 
"""

import cv2
import os
from glob import glob

# Define paths
images_folder = r"C:\Users\afiqa\OneDrive\Documents\Nurin's\FINAL YEAR PROJECT\FYP DATASETS\valid\images"
output_folder = r"C:\Users\afiqa\OneDrive\Documents\Nurin's\FINAL YEAR PROJECT\FYP DATASETS\valid\gaussian filter"

# Make sure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Define Gaussian blur parameters
kernel_size = (5, 5)  # Adjust for blurring intensity
sigma = 0  # 0 for auto-calculated sigma based on kernel size

# Define new dimensions for resizing (e.g., 256x256)
new_width = 256
new_height = 256

# Process each image
for image_path in glob(os.path.join(images_folder, '*.jpg')):  # Adjust extension if needed
    # Load the image
    image = cv2.imread(image_path)
    
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    
    # Resize the image
    resized_image = cv2.resize(blurred_image, (new_width, new_height))
    
    # Save the processed image to the output folder
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, resized_image)

print("Gaussian filter and resizing applied to all images.")

