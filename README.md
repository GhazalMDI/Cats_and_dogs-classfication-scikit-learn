🐾 Cat & Dog Classification using LBP + HOG + SVM

This project implements a complete image-classification pipeline for distinguishing cats vs. dogs using classic computer-vision techniques.
Instead of neural networks, the model uses hand-crafted features (LBP + HOG) and a powerful SVM classifier wrapped inside a Scikit-Learn Pipeline.

⭐ Features

✔ Full image preprocessing pipeline
✔ Automatic removal of low-quality images (blurry / too dark / too bright)
✔ Custom data augmentation (Flip, Rotate, Brightness, Crop)
✔ Feature extraction using:

LBP (Local Binary Pattern)

HOG (Histogram of Oriented Gradients)
✔ Image chunking for higher-resolution feature extraction
✔ SVM classifier with PCA + StandardScaler
✔ Model saving using Pickle
✔ Prediction function with image visualization

```bash
📁 Project Structure
project/
│── train.py                 # main training code
│── svm_model.pkl            # saved trained model
│── README.md                # this file
│── PetImages/               # dataset (Cats & Dogs)
│── test/                    # test images for prediction
```
🛠 Installation

All dependencies required for this project are included in the file:
```bash
requirements.txt
```

Install them using:
```bash

pip install -r requirements.txt
```
🔄 Data Processing Pipeline
1) Load & resize images

All images are converted to RGB and resized to 128×128.

2) Filter low-quality images

good_image() removes images that are:
    blurry
    too dark
    too bright
    
3) Data Augmentation

Each image generates several augmented samples:

flip
rotation
brightness shift
random crop

🧩 Feature Extraction
🔸 Local Binary Patterns (LBP)

LBP texture descriptors are extracted from each image chunk.

🔸 Histogram of Oriented Gradients (HOG)

Sobel gradients are computed and HOG histograms are extracted.

🔸 Chunking

chunk_image() splits the image into overlapping 32×32 patches (step size 16) to capture fine-grained details.

Final feature vector:

[LBP_features_of_all_chunks , HOG_features_of_all_chunks]

🧠 Model Training

The classifier is built using the following Scikit-Learn pipeline:
```bash
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=300)),
    ('svm', SVC(kernel="rbf", C=100, gamma='scale', class_weight="balanced"))
])
```
This includes:

Standardization

Dimensionality reduction (PCA)

RBF-kernel SVM classifier

📊 Training Results

Model performance on the test set:
```bash
Accuracy: 96.06%
Precision: 0.96
Recall: 0.96
F1-score: 0.96
```

Confusion Matrix:

[[3584  165]
 [ 130 3620]]

```bash
💾 Saving the Model
with open("svm_model.pkl", "wb") as f:
    pickle.dump({
        "pipeline": pipe,
        "label_encoder": label_encoder
    }, f, protocol=pickle.HIGHEST_PROTOCOL)

```
🔍 Prediction on New Images

predict_image():

loads the image

filters out low-quality images

extracts LBP + HOG features

predicts the label

displays the image with its predicted class

Example usage:
```bash
with open("svm_model.pkl", "rb") as f:
    data = pickle.load(f)

pipeline = data["pipeline"]
label_encoder = data["label_encoder"]

dataset = r"C:\Users\Ghazal\Desktop\test"
predict_image(dataset, pipeline, label_encoder)
```
🎯 Summary

This project demonstrates a complete classical computer-vision pipeline with:

Hand-crafted features (LBP + HOG)

Chunk-based high-detail feature extraction

PCA + SVM classification

High accuracy (96%)

Ready-to-use prediction script
