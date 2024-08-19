# Potato Disease Prediction Model

## Overview
This project aims to assist farmers in diagnosing diseases in their potato crops by using image classification. The model is built using a Convolutional Neural Network (CNN) architecture in TensorFlow to accurately detect and classify potato plant diseases (Early Blight, Late Blight, and Healthy).The model is trained on a dataset of potato leaf images and can be used to accurately identify the health status of potato plants. Future enhancements include deploying the model on Google Cloud Platform (GCP) and integrating it into a web application with a React frontend.


## Model Architecture
The model utilizes a Convolutional Neural Network (CNN) designed to extract features from potato leaf images and classify them into different disease categories. Key layers include:

- Convolutional Layers
- Max Pooling Layers
- Fully Connected Layers
- Dropout Layers


## Project Structure

- **Data:**
    - The dataset contains images of potato leaves categorized into three classes: Early Blight, Late Blight, and Healthy.
    - The dataset is loaded and preprocessed using TensorFlow's `image_dataset_from_directory`.
- **Model:**
    - A CNN model is constructed using TensorFlow's Keras API.
    - The model includes layers for resizing, rescaling, data augmentation, convolution, max pooling, flattening, and dense layers.
- **Training:**
    - The model is trained using the `fit` method with the training dataset.
    - Validation data is used to monitor the model's performance during training.
- **Evaluation:**
    - The trained model is evaluated on a test dataset using the `evaluate` method.
- **Prediction:**
    - The model is used to predict the class of new potato leaf images using the `predict` method.

## Steps to Run the Project

1. **Install Dependencies:**
    - Ensure that you have TensorFlow and other required libraries installed.
2. **Mount Google Drive:**
    - Mount your Google Drive to access the dataset.
3. **Load and Preprocess Data:**
    - Load the dataset from your Google Drive.
    - Preprocess the images by resizing, rescaling, and applying data augmentation.
4. **Build and Compile Model:**
    - Define the CNN model architecture.
    - Compile the model with an optimizer, loss function, and metrics.
5. **Train Model:**
    - Train the model on the training dataset.
    - Use validation data to monitor performance.
6. **Evaluate Model:**
    - Evaluate the trained model on the test dataset.
7. **Make Predictions:**
    - Use the model to predict the class of new potato leaf images.

## Results

- The model achieves high accuracy on the test dataset.
- The trained model can be used to effectively classify potato leaf diseases.

## Future Work

- Deployment: Deploy the model on GCP.
- Web Application: Integrate the model with a React frontend for user interaction.
- Model Improvements: Experiment with other architectures and techniques like transfer learning to improve accuracy.
