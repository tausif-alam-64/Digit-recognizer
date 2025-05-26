# Digit Recognition Neural Network

A simple digit recognition system built from scratch using NumPy and Matplotlib. 
This project demonstrates the implementation of a two-layer neural network trained on the popular MNIST dataset to classify handwritten digits (0–9).

## Features

- Built with only NumPy (no deep learning libraries like TensorFlow or PyTorch)
- Digit classification using a custom two-layer neural network
- ReLU and Softmax activation functions
- One-hot encoding for labels
- Accuracy tracking during training
- Option to visualize predictions
- Supports saving and loading the trained model to avoid retraining

After training, the model achieves around **85–90% accuracy** on the dev set.

## How It Works

1. **Data Preprocessing**  
   - The dataset is shuffled and split into training and dev sets.
   - Input features are normalized and transposed.

2. **Model Architecture**
   - Input Layer: 784 neurons (28x28 pixels)
   - Hidden Layer: 10 neurons, ReLU activation
   - Output Layer: 10 neurons, Softmax activation

3. **Training**
   - Uses gradient descent and backpropagation.
   - Tracks accuracy every 50 iterations.

4. **Prediction**
   - After training, the model predicts digits based on pixel input.
   - You can visualize the prediction with `matplotlib`.
