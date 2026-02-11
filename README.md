# Skin Type Classification Using CNN

A Python-based machine learning project that classifies a person’s facial skin type into **Dry**, **Normal**, or **Oily** using a Convolutional Neural Network (CNN).

## Project Overview
This project uses deep learning techniques to analyze facial images and predict skin type categories. It is designed as a simple yet practical application of computer vision and neural networks, suitable for academic and learning purposes.

The system includes:
- A CNN model for skin type classification
- A prediction script to classify new facial images
- Image preprocessing and normalization for reliable results

## Skin Type Categories
- Dry
- Normal
- Oily

## Model Training
The CNN model is trained using labeled facial images organized into class-specific folders. The training process includes:

- Image resizing to 128×128 pixels
- Pixel normalization
- Data augmentation to improve generalization
- Multiple convolutional and pooling layers
- Dropout to reduce overfitting

The trained model is saved as `skin_type_model.h5` for later use in prediction.

## Prediction Workflow
The prediction system:
1. Loads the trained CNN model
2. Preprocesses the input facial image
3. Normalizes pixel values
4. Predicts the skin type using the trained model
5. Outputs the predicted skin category

The system can also evaluate multiple images from a directory and calculate prediction accuracy.

## Project Structure
- `neural.py` – CNN model training and model saving
- `predict.py` – Image preprocessing and skin type prediction
- `predict2.py`– Image preprocessing and skin type prediction
- `skin_type_model.h5` – Trained CNN model
- `train/` – Dataset organized into class folders (dry, normal, oily)

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Computer Vision techniques
- Convolutional Neural Networks (CNN)

## Use Cases
- Skin analysis research
- Learning CNN-based image classification
- Academic mini-projects
- Computer vision experimentation

## Note
This project is intended for educational and experimental purposes and should not be used as a medical diagnostic tool.
