# Transforming Waste Management Using Transfer Learning

This project applies transfer learning to classify types of waste (e.g., trash, plastic, metal, etc.) from images. By leveraging a pre-trained deep learning model, it predicts the category of waste based on visual features, enabling smarter waste sorting systems.

## 🧠 Project Overview

The model uses a pretrained CNN architecture (e.g., ResNet or similar) and fine-tunes it on a custom waste dataset. This helps in accurately identifying waste types even with limited training data.

## 🔍 Key Features

- Image classification using PyTorch and transfer learning
- Real-time or batch prediction of waste type
- Confidence score for each prediction
- Handles image loading errors gracefully

## 📁 Directory Structure

```
.
├── predict.py                 # Main prediction script
├── model.pth                 # Trained PyTorch model (not included here)
├── dataset/                  # Folder containing image dataset
└── README.md                 # Project documentation
```

## 🖼️ Example Prediction

```bash
--- Prediction for: /path/to/image.jpg ---
Predicted Waste Type: Trash
Confidence: 0.9834
Predicted Class Index: 2
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/ashok910/waste.git
cd waste
```

### 2. Install Requirements

Ensure Python 3.7+ is installed. Then install the following dependencies:

```bash
pip install torch torchvision pillow
```

### 3. Load or Train Your Model

Place your trained model file as `model.pth` in the root directory. You can train your own or fine-tune an existing one using transfer learning.

### 4. Predict Waste Type

Update `test_image_path` in `predict.py` with your image path, and then run:

```bash
python predict.py
```

## ⚙️ Configuration

- Image size: 224x224
- Normalization values: `[0.485, 0.456, 0.406]`, `[0.229, 0.224, 0.225]` (ImageNet standard)
- Device: GPU or CPU (automatically handled in your full code)

## 🧪 Dataset

Ensure your dataset is available locally in this format:

```
TrashType_Image_Dataset/
├── metal/
├── trash/
├── plastic/
├── paper/
└── ...
```

The directory names will automatically become class labels via `ImageFolder`.

## 👨‍💻 Author

Ashok Kumar Reddy P  
[GitHub](https://github.com/ashok910)

## 📄 License

This project is licensed under the MIT License.

---

Feel free to contribute or raise issues!
