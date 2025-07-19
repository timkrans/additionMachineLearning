# Neural Network Addition Learning

A C++ implementation of a simple neural network that learns to perform addition through supervised learning. This project demonstrates the fundamentals of neural networks, backpropagation, and machine learning concepts.

## Overview

This project implements a feedforward neural network with:
- **Input Layer**: 2 neurons (for two numbers to add)
- **Hidden Layer**: 120 neurons with sigmoid activation
- **Output Layer**: 1 neuron (the predicted sum)

The network learns to add two numbers by training on thousands of random examples and uses backpropagation to adjust its weights and biases.

## Features

- **Sigmoid Activation Function**: Uses the standard sigmoid function for non-linearity
- **Backpropagation Training**: Implements gradient descent to minimize prediction error
- **Model Persistence**: Save and load trained networks to/from files
- **Normalized Inputs/Outputs**: Handles data normalization for better training
- **Random Weight Initialization**: Weights initialized randomly between -1 and 1

## How It Works

### Training Process
1. **Data Generation**: Creates random pairs of numbers (0-10) and their sums
2. **Normalization**: Scales inputs and outputs to [0,1] range for better training
3. **Forward Pass**: Computes predictions through the network
4. **Error Calculation**: Compares predictions with actual sums
5. **Backpropagation**: Updates weights and biases using gradient descent
6. **Iteration**: Repeats for 10,000 epochs

### Network Architecture
```
Input Layer (2) → Hidden Layer (120) → Output Layer (1)
```

## Building and Running

### Prerequisites
- C++ compiler (g++, clang++, or MSVC)
- Standard C++ libraries

### Compilation
```bash
g++ -std=c++11 -o addition addition.cpp
```

### Execution
```bash
./addition
```

## Output

The program will:
1. Train the neural network for 10,000 epochs
2. Save the trained model to `trained_network.txt`
3. Load the model and test it with various input pairs
4. Display predictions vs expected results

Example output:
```
Training...
Model saved to 'trained_network.txt'.

Testing loaded network:
0 + 0 = 1.87891 (expected 0)
0 + 2 = 2.79308 (expected 2)
0 + 4 = 4.08404 (expected 4)
0 + 6 = 5.78519 (expected 6)
0 + 8 = 7.83025 (expected 8)
0 + 10 = 10.0377 (expected 10)
2 + 0 = 2.80951 (expected 2)
...
```

## File Structure

- `addition.cpp` - Main source code containing the neural network implementation
- `trained_network.txt` - Saved model weights and biases (generated after training)
- `README.md` - This documentation file

## Key Classes

### Neuron
- Represents a single neuron with weights, bias, and activation function
- Handles forward propagation and weight updates
- Supports saving/loading weights

### NeuralNetwork
- Manages the complete network architecture
- Implements training with backpropagation
- Provides feedforward prediction functionality
- Handles model persistence

## Learning Parameters

- **Learning Rate**: 0.01
- **Training Epochs**: 10,000,000
- **Input Range**: 0-10 (normalized to 0-1)
- **Output Range**: 0-20 (normalized to 0-1)

## Educational Value

This project serves as an excellent learning tool for understanding:
- Neural network fundamentals
- Backpropagation algorithm
- Gradient descent optimization
- Data normalization techniques
- Model serialization
- C++ object-oriented programming

## Limitations

- Simple architecture (single hidden layer)
- Limited to addition of two numbers
- Fixed learning parameters
- No validation set or early stopping
- Basic error handling

## Future Enhancements

Potential improvements could include:
- Support for more mathematical operations
- Multiple hidden layers
- Different activation functions
- Adaptive learning rates
- Cross-validation
- Better error handling
- GUI interface for testing
