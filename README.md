# Lemon Leaf Disease Detection (Ensemble CNN + Grad-CAM)

This repository contains an end-to-end **lemon leaf disease classification** project using multiple CNN architectures and a **weighted ensemble learning approach**, along with **Explainable AI (XAI)** techniques such as **Grad-CAM** and **Ablation-CAM** to interpret model predictions.

The complete workflow is implemented in the notebook:  
`lemon_leaf_disease_dataset_for_advanced_detection.ipynb`

---

## ✨ Key Features

- Image classification using **PyTorch** and **Torchvision**
- Multiple CNN backbones:
  - **ResNet50**
  - **EfficientNet-B0**
  - **MobileNetV2**
  - **DenseNet121**
- **Weighted Ensemble Model**
  - Learns optimal weights for combining model predictions
- Model evaluation using standard classification metrics
- **Explainable AI (XAI)**
  - **Grad-CAM** visualization
  - **Ablation-CAM** visualization

---

## 🗂️ Dataset

The dataset used in this project is publicly available on **Mendeley Data**:

🔗 **Dataset Link:**  
https://data.mendeley.com/datasets/44nrn4593f/1

### Dataset Description
- Contains images of **lemon leaves** affected by different diseases
- Includes both **healthy** and **diseased** leaf samples
- Suitable for image classification and plant disease detection tasks

### Classes
The dataset includes the following classes:
- Bacterial Blight
- Canker
- Curl Virus
- Healthy

---

## 📁 Dataset Structure

The dataset is organized in a directory structure compatible with  
`torchvision.datasets.ImageFolder`.
```
.data/
├── train/
│ ├── Bacterial_Blight/
│ ├── Canker/
│ ├── Curl_Virus/
│ └── Healthy/
├── val/
│ ├── Bacterial_Blight/
│ ├── Canker/
│ ├── Curl_Virus/
│ └── Healthy/
└── test/
├── Bacterial_Blight/
├── Canker/
├── Curl_Virus/
└── Healthy/

## 🧪 Methodology

The project follows a structured deep learning workflow:

1. **Data Loading**
   - Images are loaded using `torchvision.datasets.ImageFolder`
   - Dataset is split into training, validation, and test sets

2. **Preprocessing & Augmentation**
   - Image resizing and normalization
   - Data augmentation applied to improve generalization

3. **Model Training**
   - Trained multiple CNN architectures:
     - ResNet50
     - EfficientNet-B0
     - MobileNetV2
     - DenseNet121

4. **Ensemble Learning**
   - Combined predictions from individual models
   - Learned optimal ensemble weights for final prediction

5. **Evaluation**
   - Evaluated performance using classification accuracy and related metrics

6. **Explainable AI (XAI)**
   - Applied Grad-CAM to visualize discriminative regions
   - Used Ablation-CAM for more robust interpretability


