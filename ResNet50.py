# -*- coding: utf-8 -*-
"""

@author: afiqa
"""
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to your positive and negative folders for train, validation, and test sets
train_dir = r"C:\Users\User\OneDrive\Documents\Nurin's\FINAL YEAR PROJECT\FYP DATASET\ResNet50 gaussian filter\train"
valid_dir = r"C:\Users\User\OneDrive\Documents\Nurin's\FINAL YEAR PROJECT\FYP DATASET\ResNet50 gaussian filter\valid"
test_dir = r"C:\Users\User\OneDrive\Documents\Nurin's\FINAL YEAR PROJECT\FYP DATASET\ResNet50 gaussian filter\test"

# Data generators with normalization
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
valid_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Image loading with class_mode set for binary classification
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Load ResNet50 model, excluding the top layer
base_model = tf.keras.applications.ResNet50(
    weights='imagenet', 
    include_top=False, 
    input_shape=(224, 224, 3)
)

# Freeze the base model
base_model.trainable = False

# Add custom layers for binary classification
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary output
])

# Compile the model
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=100
)

# Predict the classes on test data
test_labels = test_generator.classes
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int).ravel()

# Confusion matrix and classification report
cm = confusion_matrix(test_labels, predicted_classes)
print("Confusion Matrix:")
print(cm)

# Plotting the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(test_labels, predicted_classes, target_names=["Negative", "Positive"]))
