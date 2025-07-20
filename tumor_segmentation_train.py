# train_unet.py – Brain Tumor Segmentation using U-Net

import os
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Dataset path
IMG_HEIGHT, IMG_WIDTH = 128, 128
IMAGE_DIR = "data/images/"       # Folder with MRI images
MASK_DIR = "data/masks/"         # Folder with binary masks

# Load and preprocess images and masks
def load_images_and_masks(image_dir, mask_dir):
    image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
    mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))

    images, masks = [], []
    for img_path, mask_path in zip(image_paths, mask_paths):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
        mask = np.expand_dims(mask, axis=-1)
        images.append(img)
        masks.append(mask)
    return np.array(images), np.array(masks)

X, y = load_images_and_masks(IMAGE_DIR, MASK_DIR)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# U-Net model
def build_unet(input_size=(128, 128, 3)):
    inputs = Input(input_size)
    c1 = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(64, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D()(c3)

    c4 = Conv2D(128, 3, activation='relu', padding='same')(p3)
    c4 = Conv2D(128, 3, activation='relu', padding='same')(c4)

    u1 = Conv2DTranspose(64, 2, strides=2, padding='same')(c4)
    u1 = concatenate([u1, c3])
    c5 = Conv2D(64, 3, activation='relu', padding='same')(u1)
    c5 = Conv2D(64, 3, activation='relu', padding='same')(c5)

    u2 = Conv2DTranspose(32, 2, strides=2, padding='same')(c5)
    u2 = concatenate([u2, c2])
    c6 = Conv2D(32, 3, activation='relu', padding='same')(u2)
    c6 = Conv2D(32, 3, activation='relu', padding='same')(c6)

    u3 = Conv2DTranspose(16, 2, strides=2, padding='same')(c6)
    u3 = concatenate([u3, c1])
    c7 = Conv2D(16, 3, activation='relu', padding='same')(u3)
    c7 = Conv2D(16, 3, activation='relu', padding='same')(c7)

    outputs = Conv2D(1, 1, activation='sigmoid')(c7)
    model = Model(inputs, outputs)
    return model

model = build_unet()
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=8)

# Save
model.save("unet_brain_tumor.h5")
print("✅ U-Net model saved.")
