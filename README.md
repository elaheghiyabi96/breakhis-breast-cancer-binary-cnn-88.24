# breakhis-breast-cancer-binary-cnn-88.24
Binary classification of breast cancer histopathology images (Benign vs Malignant) using an improved CNN on the BreakHis dataset - 88.24% test accuracy.
# 🧬 BreakHis Breast Cancer Binary Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![Keras](https://img.shields.io/badge/Keras-2.10+-red.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-88.24%25-brightgreen.svg)
![Dataset](https://img.shields.io/badge/Dataset-BreakHis-9cf.svg)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Supported-yellow.svg)

**CNN-based binary classification of breast cancer histopathology images (Benign vs Malignant)**

</div>

---

## 📌 Overview

This project implements an **improved Convolutional Neural Network (CNN)** for binary classification of breast cancer histopathology images using the **BreakHis** dataset. The model distinguishes between **Benign** and **Malignant** breast tumor tissues with **88.24% test accuracy**, significantly outperforming the baseline model.

---

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **88.24%** |
| **Benign Precision** | 0.87 |
| **Benign Recall** | 0.73 (↑ 20% from baseline) |
| **Malignant Precision** | 0.89 |
| **Malignant Recall** | 0.95 |
| **F1-Score (Weighted)** | 0.88 |
| **Model Size** | 37.6 MB |

---

## 📊 Dataset Overview

The **BreakHis** (Breast Cancer Histopathological Images) dataset consists of microscopic images of breast tumor tissues.

| Feature | Description |
|---------|-------------|
| **Total Images** | 7,909 |
| **Classes** | Benign (2,480) / Malignant (5,429) |
| **Magnifications** | 40X, 100X, 200X, 400X |
| **Image Size** | 224 × 224 × 3 (RGB) |
| **Format** | PNG / JPG / JPEG |

### 📈 Class Distribution

```
Benign    ████████████████░░░░░░░░░░░░░░░░  2,480 (31.4%)
Malignant ████████████████████████████████  5,429 (68.6%)
```

⚠️ **Note:** The dataset is imbalanced (Malignant = 69%, Benign = 31%). Class-specific augmentation was applied to address this.

---

## 🧠 Model Architecture

### Improved CNN Design

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 222, 222, 32)   │           896 │
│ max_pooling2d (MaxPooling2D)    │ (None, 111, 111, 32)   │             0 │
│ conv2d_1 (Conv2D)               │ (None, 109, 109, 64)   │        18,496 │
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 54, 54, 64)     │             0 │
│ conv2d_2 (Conv2D)               │ (None, 52, 52, 128)    │        73,856 │
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 26, 26, 128)    │             0 │
│ conv2d_3 (Conv2D)               │ (None, 24, 24, 256)    │       295,168 │  ← New layer
│ max_pooling2d_3 (MaxPooling2D)  │ (None, 12, 12, 256)    │             0 │
│ flatten (Flatten)               │ (None, 36864)          │             0 │
│ dense (Dense)                   │ (None, 256)            │     9,437,440 │  ← Increased
│ dropout (Dropout)               │ (None, 256)            │             0 │
│ dense_1 (Dense)                 │ (None, 128)            │        32,896 │
│ dropout_1 (Dropout)             │ (None, 128)            │             0 │
│ dense_2 (Dense)                 │ (None, 1)              │           129 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
```

**Total Parameters:** 9,858,881 (37.6 MB)

### Key Improvements Over Baseline

| Improvement | Details |
|-------------|---------|
| **4th Conv Layer** | Added 256 filters for deeper feature extraction |
| **Increased Dense Units** | 256 → 512 → 128 for better representation |
| **Class-Specific Augmentation** | Stronger augmentation for Benign class |
| **Learning Rate Scheduling** | Automatic reduction on plateau |
| **Early Stopping** | Prevent overfitting |

---

## 🔧 Implementation Details

### Data Augmentation

```python
# Benign (minority class) - Strong augmentation
benign_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

# Malignant (majority class) - Light augmentation
malignant_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True
)
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Batch Size** | 32 |
| **Epochs** | 30 |
| **Learning Rate** | 1e-4 (with ReduceLROnPlateau) |
| **Optimizer** | Adam |
| **Loss Function** | Binary Cross-Entropy |
| **Train/Test Split** | 80% / 20% |
| **Patience (Early Stopping)** | 7 |
| **Patience (Reduce LR)** | 3 |

---

## 📈 Results

### Classification Report

```
              precision    recall  f1-score   support
      Benign       0.87      0.73      0.80       496
   Malignant       0.89      0.95      0.92      1086

    accuracy                           0.88      1582
   macro avg       0.88      0.84      0.86      1582
weighted avg       0.88      0.88      0.88      1582
```

### Confusion Matrix

```
[[ 362  134]   ← Benign: 362 correct, 134 misclassified
 [  52 1034]]  ← Malignant: 1034 correct, 52 misclassified
```

### Performance Comparison

| Metric | Baseline | **Improved Model** |
|--------|----------|-------------------|
| **Accuracy** | 83% | **88.24%** ✅ |
| **Benign Recall** | 53% | **73%** ✅ (+20%) |
| **Malignant Precision** | 82% | **89%** ✅ (+7%) |
| **Parameters** | 11.1M | **9.9M** (↓ 11%) |
| **Model Size** | 42.6 MB | **37.6 MB** (↓ 12%) |

---

## 🚀 How to Run

### 1. 📥 Setup Google Colab

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

### 2. 📂 Dataset Preparation

Place your dataset zip file in Google Drive with this path:
```
/content/drive/MyDrive/dataset_cancer_v1.zip
```

**Expected folder structure:**
```
dataset_cancer_v1/
└── classificacao_binaria/
    ├── 40X/
    │   ├── benign/
    │   └── malignant/
    ├── 100X/
    │   ├── benign/
    │   └── malignant/
    ├── 200X/
    │   ├── benign/
    │   └── malignant/
    └── 400X/
        ├── benign/
        └── malignant/
```

### 3. 🚀 Run the Code

Simply execute all cells in the Colab notebook. The code will:
- ✅ Extract the dataset automatically
- ✅ Preprocess images
- ✅ Train the improved CNN model
- ✅ Generate evaluation metrics
- ✅ Save the model as `.keras` file
- ✅ Plot training history

### 4. 📊 Outputs

After training, you'll receive:
- 📊 **Classification Report**
- 📊 **Confusion Matrix**
- 📈 **Training/Validation Curves**
- 💾 **Saved Model** (`breast_cancer_cnn_binary.keras`)

---

## 🛠️ Requirements

Install dependencies with:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
```

### Dependencies

```
tensorflow>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
```

---

## 📁 Repository Structure

```
breakhis-breast-cancer-binary-cnn/
│
├── README.md                                    ← You are here
├── requirements.txt                             ← Dependencies
├── breast_cancer_binary_cnn_improved.py         ← Main training script
├── breast_cancer_binary_cnn_improved.ipynb      ← Colab notebook
├── training_history.png                         ← Training curves
└── .gitignore
```

---

## 💡 How to Use the Trained Model

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = tf.keras.models.load_model('breast_cancer_cnn_binary.keras')

# Load and preprocess image
img = image.load_img('sample.jpg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)[0][0]
result = "Malignant" if prediction > 0.5 else "Benign"
print(f"Prediction: {result} (confidence: {prediction:.2f})")
```

---

## 🔍 Interpretation of Results

| What It Means | Details |
|---------------|---------|
| **High Sensitivity (Malignant 95%)** | Model rarely misses true malignant cases → Good for screening |
| **Improved Benign Recall (73%)** | Better at identifying benign cases than baseline (20% improvement) |
| **Precision Score (89%)** | Low false positive rate → Trustworthy predictions |
| **Consistent F1-Score (0.88)** | Balanced performance across both classes |

---

## 🎓 Lessons Learned

1. **Class-Specific Augmentation Works**  
   - Stronger augmentation for minority class (Benign) significantly improved recall

2. **Deeper Architecture Helps**  
   - Adding a 4th Conv layer (256 filters) extracted better features

3. **Learning Rate Scheduling**  
   - Automatic LR reduction prevented overshooting and improved convergence

4. **Early Stopping**  
   - Prevented overfitting when training accuracy reached 96%

5. **Managing Class Imbalance**  
   - Even with imbalance (69% Malignant), model performed well on both classes

---

## ⚠️ Limitations

- ❌ **No Transfer Learning** – Could potentially improve accuracy further
- ❌ **Patient-Level Split** – Currently uses image-level split (may cause data leakage)
- ❌ **No Cross-Validation** – Only single train/test split
- ❌ **Limited Data** – More images could improve robustness
- ❌ **No Hyperparameter Tuning** – Could optimize further

---

## 🚀 Future Improvements

| Priority | Improvement | Expected Impact |
|----------|-------------|-----------------|
| 1 | Transfer Learning (EfficientNet/ResNet) | +3-5% accuracy |
| 2 | Patient-wise Cross-Validation | More reliable evaluation |
| 3 | Hyperparameter Tuning | Optimize learning rate, layers |
| 4 | Ensemble Methods | Combine multiple models |
| 5 | Grad-CAM Visualization | Explain model decisions |
| 6 | Multiclass Classification | 8 tumor subtypes |

---

## 📚 Dataset Citation

If you use this code, please cite the original BreakHis dataset:

```
@article{breakhis2016,
  title={BreakHis: A Dataset for Breast Cancer Histopathological Image Classification},
  author={Spanhol, F.A. and Oliveira, L.S. and Petitjean, C. and Heutte, L.},
  journal={IEEE Transactions on Biomedical Engineering},
  year={2016}
}
```

---

## 🌐 Dataset Source

**🔗 BreakHis Dataset on Mendeley Data:**  
[https://data.mendeley.com/datasets/jxwvdwhpc2/1](https://data.mendeley.com/datasets/jxwvdwhpc2/1)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ If this project helped you, don't forget to star it! ⭐**

</div>
