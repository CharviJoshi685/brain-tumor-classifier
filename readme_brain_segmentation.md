# 🧠 Brain Tumor Segmentation with U-Net

This project uses a **U-Net Convolutional Neural Network** for segmenting brain tumors in MRI images. A trained deep learning model identifies tumor regions and overlays a predicted mask on top of the input image.

---

## 🚀 Features

- Upload MRI scans
- View tumor segmentation mask
- Overlay prediction on input image
- Built with TensorFlow + Streamlit

---

## 🧠 Deep Learning Architecture

- **Model:** U-Net
- **Input Size:** 128x128 RGB MRI scans
- **Loss Function:** Binary Crossentropy
- **Output:** Binary mask (tumor or not)

---

## 📁 File Structure

```
├── train_unet.py          # Training script for U-Net
├── app.py                 # Streamlit app for segmentation
├── unet_brain_tumor.h5    # Trained model weights
├── data/images/           # Folder for training MRI images
├── data/masks/            # Folder for training binary masks
├── requirements.txt
├── README.md
├── .gitignore
```

---

## 📊 Dataset

Use any public brain MRI tumor segmentation dataset. Suggested:

- [BRATS Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- Format: `.png` images & masks of the same size

---

## ⚙️ Installation & Usage

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/brain-tumor-unet.git
cd brain-tumor-unet
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model (optional)

```bash
python train_unet.py
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 🧪 Example

Upload a brain MRI image → Click "Segment Tumor" → View predicted overlay & binary mask

---

## 📄 License

This project is open-source and licensed under the MIT License.

---

## 🙋‍♂️ Author

Built by [Your Name] with 🧠 using U-Net and Python

