

# Handwriting-to-Text Recognition Model (C Implementation)

This repository contains the C implementation of a Convolutional Neural Network (CNN) model trained on the MNIST dataset for recognizing handwritten digits.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/handwriting-to-text-recognition-c.git
   ```

2. Compile the code:
   ```bash
    make
   ```

## Usage

1. Run the compiled executable:
   ```bash
   ./main
   ```

## Model Architecture

The model architecture is implemented in `model.c` and consists of functions for initializing the model, performing convolution operations, and applying activation functions.

## Training

The training data is loaded from the MNIST dataset and the model is trained using stochastic gradient descent.

## Evaluation

The model's performance is evaluated on the test dataset to measure its accuracy in recognizing handwritten digits.

