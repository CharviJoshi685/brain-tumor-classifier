# app.py – Brain Tumor Segmentation Viewer

import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="🧠 Brain Tumor Segmentation", layout="centered")
st.title("🧠 Brain Tumor Segmentation with U-Net")
st.write("Upload an MRI scan to visualize predicted tumor region.")

model = load_model("unet_brain_tumor.h5")
IMG_SIZE = 128

def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    return arr, img

def predict_segmentation(image_array):
    img_input = np.expand_dims(image_array, axis=0)
    pred_mask = model.predict(img_input)[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8).squeeze()
    return pred_mask

uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image_array, original_img = preprocess_image(uploaded_file)
    st.image(original_img, caption="Uploaded MRI", use_column_width=True)

    if st.button("Segment Tumor"):
        mask = predict_segmentation(image_array)
        mask_color = np.stack([mask*255]*3, axis=-1)
        overlay = cv2.addWeighted(np.array(original_img), 0.7, mask_color.astype(np.uint8), 0.3, 0)

        st.image(overlay, caption="Segmented Tumor Overlay", use_column_width=True)
        st.image(mask*255, caption="Predicted Mask", use_column_width=True)
