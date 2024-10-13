# Handwriting-to-Text Recognition Model

This project implements a Convolutional Neural Network (CNN) for handwritten digit recognition using the MNIST dataset. The model is implemented in both C and Python, allowing for a comparison of performance between the two languages.

## Table of Contents

- [Dataset Description](#dataset-description)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)
- [Future Scope](#future-scope)
- [License](#license)

## Dataset Description

The MNIST dataset consists of 60,000 training examples and 10,000 test examples of handwritten digits. The digits are size-normalized and centered within a fixed-size 28x28 pixel image.

You can find more information about the dataset [here](http://yann.lecun.com/exdb/mnist/).

## Model Architecture

The CNN architecture consists of the following layers:

1. **Input Layer**: 1x28x28 (1 channel, 28x28 pixels)
2. **Convolutional Layer 1 (Conv1)**: 16 filters, output size 16x14x14, kernel size 3x3, padding=1, stride=2
3. **Convolutional Layer 2 (Conv2)**: 32 filters, output size 32x7x7, kernel size 3x3, padding=1, stride=2
4. **Fully Connected Layer 1 (FC1)**: 200 nodes
5. **Fully Connected Layer 2 (FC2)**: 200 nodes
6. **Output Layer**: 10 nodes with Softmax activation function

The model architecture can be modified in the following function of `model_saver.c`:

```c
void init_model_architecture(layer_component **linput, layer_component **lconv1, layer_component **lconv2,
                             layer_component **lfull1, layer_component **lfull2, layer_component **loutput)
{
    // Input layer - 1x28x28.
    *linput = create_input_layer(1, 28);
    // Conv1 layer - 16x14x14, 3x3 conv, padding=1, stride=2. kernel - 3
    *lconv1 = create_conv_layer(*linput, 16, 14, 3, 1, 2, 0.1);
    // Conv2 layer - 32x7x7, 3x3 conv, padding=1, stride=2, kernel - 3
    *lconv2 = create_conv_layer(*lconv1, 32, 7, 3, 1, 2, 0.1);
    // FC1 layer - 200 nodes.
    *lfull1 = create_full_layer(*lconv2, 200, 0.1);
    // FC2 layer - 200 nodes.
    *lfull2 = create_full_layer(*lfull1, 200, 0.1); // Fully connected layer - 200 nodes. 
    *loutput = create_full_layer(*lfull2, 10, 0.1); // Output layer - 10 nodes.
}
```

## Training

The model was trained using the following parameters:
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Learning Rate**: 0.1
- **Loss Function**: Mean Squared Error
- **Batch Size**: 64
- **Epochs**: 10

The training data was split into training and validation sets to prevent overfitting.

## Results

After training, the model achieved the following accuracies:

| Implementation       | Training Accuracy | Test Accuracy |
|----------------------|-------------------|---------------|
| C Implementation      | 98.61%            | 97.42%        |
| PyTorch Implementation | 99.49%            | 98.22%        |

The model parameters can be saved and loaded in `.txt` format, stored in `./results/model.txt`.

### Model Architecture Visualization

![Model Architecture](assets/model_architecture.png)

### Training and Validation Accuracy

![Training and Validation Accuracy](assets/training_validation_accuracy.png)

## Usage

To run the model:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/handwriting-to-text-recognition.git
   cd handwriting-to-text-recognition
   ```

2. Compile the C implementation:
   ```bash
   gcc -o cnn cnn.c model_saver.c -lm
   ```

3. Run the C implementation:
   ```bash
   ./cnn
   ```

4. For the Python implementation, ensure you have the required libraries installed:
   ```bash
   pip install torch torchvision
   ```

5. Run the Python script:
   ```bash
   python train.py
   ```

## Future Scope

Several avenues for exploration still exist to achieve better performance:
- Hyperparameter optimization for fine-tuning the current CNN architecture.
- Implementing data augmentation for improved robustness.
- Incorporating binary cross-entropy loss and integrating Softmax activation.
- Extending the model to recognize handwritten characters, complete words, or sentences using additional convolutional layers and Recurrent Neural Networks (RNNs).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
