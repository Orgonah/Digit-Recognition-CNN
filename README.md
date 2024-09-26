# Digit-Recognition-CNN

This project demonstrates a Convolutional Neural Network (CNN) model built using Keras for classifying handwritten digits from the MNIST dataset. The model achieves an accuracy of 99.15%.

## Overview

The MNIST dataset is a classic dataset of handwritten digits used for training image processing systems. It contains 60,000 training samples and 10,000 testing samples. Each image is a 28x28 pixel grayscale image.

## Model Architecture

The CNN model consists of the following layers:
- Input layer: 28x28x1 grayscale images
- 1st Convolutional layer: 32 filters, 3x3 kernel, ReLU activation
- 1st MaxPooling layer: 2x2 pool size
- 2nd Convolutional layer: 64 filters, 3x3 kernel, ReLU activation
- 2nd MaxPooling layer: 2x2 pool size
- Flatten layer
- Dropout layer: 50% dropout rate
- Dense layer: 10 neurons, softmax activation

## Results

- Test Loss: 0.024
- Test Accuracy: 99.15%

