# Handwriting_emnist-mnist_prediction_using_CNN_model
This repository contains a simple, yet complete, implementation of a Convolutional Neural Network (CNN) built with PyTorch for image classification. The model is trained to recognize handwritten letters (A-Z) using the EMNIST 'letters' dataset.

The primary purpose of this project is to serve as a clear, self-contained template for the complete PyTorch workflow—from data loading and preprocessing to model definition, training, and robust evaluation.

Key Features
1. Robust Data Pipeline
EMNIST 'Letters' Dataset: Explicitly uses the EMNIST letters split, providing 26 classes (A-Z).

Standard Transforms: Applies transforms.ToTensor() and transforms.Normalize() for consistent image preprocessing.

Crucial Label Adjustment: Addresses the non-standard 1-indexed labeling (1-26) of the EMNIST 'letters' dataset by performing a label shift (labels = labels - 1) in both the training and testing loops. This is a critical step for compatibility with PyTorch's nn.CrossEntropyLoss, which expects 0-indexed labels (0-25).

2. Simple CNN Architecture
The model (CNNModel) is defined as a standard nn.Module subclass.

Convolutional Layers: Two layers of 3x3 convolutions with ReLU activation and Max Pooling.

Pooling: nn.MaxPool2d(2, 2) for feature map reduction.

Classification Head: Two Fully Connected (Linear) layers, outputting 26 nodes for the final A-Z classification.

3. Training & Evaluation
Loss Function: nn.CrossEntropyLoss (combines softmax and negative log-likelihood loss).

Optimizer: optim.Adam with a learning rate of 0.001.

Complete Loops: Includes a three-epoch training loop to monitor loss and a separate evaluation loop to calculate final test accuracy 

The intended accuracy of this project, once the critical label-shifting bug is corrected, is approximately 92.70%.
