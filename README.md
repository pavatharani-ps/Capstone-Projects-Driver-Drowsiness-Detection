🚗 Driver Drowsiness Detection using Deep Learning
📌 Project Overview

This project detects driver fatigue using facial images and deep learning techniques.
The system analyzes eye closure and drowsiness-related facial features to classify whether the driver is:

✅ Alert
⚠️ Fatigued

The project uses Transfer Learning with MobileNetV2 and is implemented using PyTorch and Streamlit.

🎯 Objectives
Detect driver drowsiness using computer vision
Improve road safety
Build a non-intrusive fatigue monitoring system
Analyze fatigue progression over time

🧠 Technologies Used
Python
PyTorch
Torchvision
MobileNetV2
Streamlit
Matplotlib
Scikit-learn

📂 Dataset Structure
dataset/
│
├── alert/
│   ├── img1.jpg
│   ├── img2.jpg
│
├── drowsy/
│   ├── img1.jpg
│   ├── img2.jpg

🔍 Project Workflow
Step 1 — Problem Understanding

Driver fatigue is a major reason for road accidents.
This system detects drowsiness using facial indicators such as:

Eye closure
Yawning behavior
Step 2 — Dataset Collection

Collected facial images for:

Eyes Open
Eyes Closed
Yawning
No Yawn
Step 3 — Dataset Organization

Dataset split into:

Training Set
Validation Set
Test Set
Step 4 — Data Preprocessing

Applied preprocessing techniques:

Resize images to 224×224
Normalize images
Data augmentation
Rotation
Brightness variation
Horizontal flipping
Step 5 — Model Selection

Used:

MobileNetV2 Transfer Learning Model

Benefits:

Faster training
Better accuracy
Lightweight architecture
Step 6 — Model Development

Modified MobileNetV2 classifier layer for binary classification:

Alert
Fatigue
Step 7 — Model Training

Training configuration:

Optimizer: Adam
Loss Function: CrossEntropyLoss
Epochs: 5
Step 8 — Model Evaluation

Evaluation metrics:

Accuracy
Confusion Matrix
Precision
Recall


Step 9 — Decision Fusion Logic

Predictions converted into fatigue stages:
| Prediction | Fatigue Level |
| ---------- | ------------- |
| Alert      | Normal        |
| Fatigue    | Drowsy        |


Step 10 — Fatigue Progression Curve

Generated a fatigue progression graph over time to analyze fatigue evolution during driving.

📊 Output

The system predicts:

✅ ALERT
⚠️ FATIGUE DETECTED

It also displays:

Accuracy
Confusion matrix
Fatigue progression curve
