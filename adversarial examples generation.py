# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 00:32:58 2025

@author:nurin afiqah
"""

import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model

# Load the pre-trained model
model = MobileNetV2(weights='imagenet')
model.trainable = False

# FGSM function to generate adversarial examples
def create_adversarial_pattern(model, input_image, input_label, epsilon):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = tf.keras.losses.sparse_categorical_crossentropy(input_label, prediction)

    # Get the gradients of the loss w.r.t. the input image
    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    adversarial_image = input_image + epsilon * signed_grad
    adversarial_image = tf.clip_by_value(adversarial_image, -1, 1)
    return adversarial_image

# Function to process and replace adversarial examples
def generate_adversarial_images(input_dir, output_dir, epsilon=0.01):
    for root, _, files in os.walk(input_dir):  # Recursively iterate through subfolders
        for img_name in files:
            img_path = os.path.join(root, img_name)
            try:
                # Load and preprocess the image
                img = load_img(img_path, target_size=(224, 224))
                img_array = img_to_array(img)
                img_array = preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0)

                # Get the model's prediction
                prediction = model.predict(img_array)
                predicted_label = np.argmax(prediction)

                # Generate adversarial example
                adversarial_image = create_adversarial_pattern(
                    model, tf.convert_to_tensor(img_array), tf.convert_to_tensor([predicted_label]), epsilon
                )

                # Save the adversarial image
                adversarial_image = tf.squeeze(adversarial_image).numpy()
                adversarial_image = ((adversarial_image + 1.0) * 127.5).astype(np.uint8)  # Convert back to [0, 255]

                # Replicate folder structure in output directory
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, "adversarial_examples", relative_path)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                output_path = os.path.join(output_subdir, f"adv_{img_name}")
                tf.keras.preprocessing.image.save_img(output_path, adversarial_image)
                print(f"Adversarial image replaced: {output_path}")

            except Exception as e:
                print(f"Failed to process image {img_name} in {root}: {e}")

# Input and output directories
input_directory = r"C:\Users\afiqa\OneDrive\Documents\Nurin's\FINAL YEAR PROJECT\FYP DATASETS\structured canny edge"  # Replace with your input folder path
output_directory = r"C:\Users\afiqa\OneDrive\Documents\Nurin's\FINAL YEAR PROJECT\FYP DATASETS\adversarial examples of canny edge"  # Replace with your output folder path

# Replace adversarial images with higher epsilon value
generate_adversarial_images(input_directory, output_directory, epsilon=0.1)
