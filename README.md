# Plant Disease Detection: Image Processing Using Convolutional Neural Networks (CNN)

## Overview

Plant diseases cause significant agricultural losses, reducing yields by up to 40%. This project leverages deep learning to classify plant leaves into healthy and diseased categories across 38 classes. Using a combination of a custom Convolutional Neural Network (CNN) and the pre-trained MobileNet model, the project demonstrates the potential of machine learning in automating disease detection.

The system uses the **PlantVillage dataset** with over 160,000 labeled images, aiming to assist farmers and agricultural consultants by providing an easy-to-use, efficient, and accurate disease diagnosis tool.

---

## Key Features
- **Custom CNN Model:** Lightweight architecture for quick training and inference.
- **MobileNet Model:** Pre-trained CNN using transfer learning for higher accuracy.
- **Flask Web App:** Upload an image to classify diseases with detailed confidence scores.
- **Data Augmentation:** Techniques like rotation, flipping, and zooming for better generalization.
- **Comparative Analysis:** Evaluation of accuracy, training speed, and robustness of both models.

---

## Dataset
The project utilizes the **PlantVillage dataset**, which contains labeled images of both healthy and diseased leaves from 38 crop species, including :
- Tomato
- Potato
- Apple
- Corn

The dataset was preprocessed to include:
1. **Image Resizing:**  
   - Custom CNN: 150x150 pixels  
   - MobileNet: 224x224 pixels  
2. **Normalization:** Pixel values rescaled to [0, 1].
3. **Augmentation:** Random rotations, flips, and shifts to enhance model generalization.

---

## Methodology

### 1. **Custom CNN**
- **Architecture:** 3 convolutional layers, followed by max pooling, dropout, and dense layers.
- **Optimizer:** Adam (learning rate = 0.001)
- **Accuracy:**  
  - Training: 85.61%  
  - Validation: 91.97%  

### 2. **MobileNet (Pre-trained)**
- **Transfer Learning:** Fine-tuned for plant disease classification.
- **Optimizer:** Adam (learning rate = 0.001)
- **Accuracy:**  
  - Training: 87.82%  
  - Validation: 93.64%

### Deployment
The trained models were integrated into a **Flask-based web application**, allowing users to upload leaf images for disease classification. Predictions include the disease class and confidence scores.

---

## Results
| Model        | Training Accuracy | Validation Accuracy | Validation Loss |
|--------------|-------------------|---------------------|-----------------|
| Custom CNN   | 85.61%           | 91.97%             | 0.2524          |
| MobileNet    | 87.82%           | 93.64%             | 0.1527          |

**Key Observations:**
- MobileNet outperformed Custom CNN in accuracy and robustness.
- Custom CNN is faster and computationally lighter, suitable for edge deployment.

---

## Requirements

### Hardware
- **Minimum:** Intel Core i5, 8GB RAM, NVIDIA Tesla K80 GPU
- **Recommended:** Intel Core i7/Ryzen 7, 16GB RAM, NVIDIA RTX 3080 GPU

### Software
- Python 3.9+
- TensorFlow, Keras, Flask
- Libraries: NumPy, Pandas, OpenCV, Matplotlib

---

## How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/VipinYadav16/Multimodel-Plant-Disease-Detection.git
   cd Multimodel-Plant-Disease-Detection
