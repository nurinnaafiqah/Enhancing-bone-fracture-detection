# -*- coding: utf-8 -*-
"""

@author: afiqa
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Data generators
train_dir = r"C:\Users\afiqa\OneDrive\Documents\Nurin's\FINAL YEAR PROJECT\FYP DATASETS\adversarial examples of canny edge\adversarial_examples\train"
valid_dir = r"C:\Users\afiqa\OneDrive\Documents\Nurin's\FINAL YEAR PROJECT\FYP DATASETS\adversarial examples of canny edge\adversarial_examples\valid"
test_dir = r"C:\Users\afiqa\OneDrive\Documents\Nurin's\FINAL YEAR PROJECT\FYP DATASETS\adversarial examples of canny edge\adversarial_examples\test"

# Data preprocessing
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

valid_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

# Data loading
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=8,
    class_mode="categorical",
)

valid_generator = valid_test_datagen.flow_from_directory(
    valid_dir,
    target_size=(224, 224),
    batch_size=8,
    class_mode="categorical",
)

test_generator = valid_test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=1,
    class_mode="categorical",
    shuffle=False,
)

# Function to build U-Net model with pre-trained EfficientNetB0 backbone
def build_unet_classifier(input_shape, n_classes):
    inputs = Input(shape=input_shape)

    # Use EfficientNetB0 as the encoder
    backbone = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=inputs)

    # Collect the output of specific blocks for skip connections
    skips = [backbone.get_layer(name).output for name in ['block1a_activation', 'block2b_expand_activation', 'block3b_expand_activation', 'block4c_expand_activation']]

    # Decoder
    x = backbone.output
    for skip in reversed(skips):
        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, skip])
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    # Custom classification head
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model


# Model parameters
input_shape = (224, 224, 3)
n_classes = train_generator.num_classes  # Dynamically infer number of classes

# Build the model
model = build_unet_classifier(input_shape=input_shape, n_classes=n_classes)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('unet_efficientnet_model.keras', save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    epochs=15,  # Adjust as needed
    validation_data=valid_generator,
    callbacks=[checkpoint, early_stopping]
)

# Evaluate the model
test_labels = test_generator.classes
predictions = model.predict(test_generator)
predicted_classes = predictions.argmax(axis=1)  # Use argmax for multi-class predictions

# Confusion matrix and metrics
cm = confusion_matrix(test_labels, predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(test_labels, predicted_classes, target_names=test_generator.class_indices.keys()))

# Plot training and validation loss/accuracy
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
