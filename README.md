# Digit Recognizer

The Digit Recognizer project aims to accurately identify handwritten digits using the **MNIST** dataset. This dataset consists of tens of thousands of images of handwritten digits, which serves as a classic benchmark for machine learning algorithms.

## Import Required Dependencies


1. **PyTorch**: For tensor operations and neural network functionalities. Includes `Torchvision` for image processing, dataset handling, and transformations.
2. **Pandas**: For data manipulation and saving results to CSV.
3. **Matplotlib**: For visualizing images and results.

```python
pip install -r requirements.txt
```

## Load & Prepare The Dataset

The **MNIST** dataset, a widely used benchmark in computer vision, is utilized for training and testing the model.
The data is transformed and normalized before being loaded into DataLoaders.

### Data Transformation

1. Applies a series of image transformations.
2. Converts images to PyTorch tensors.
3. Normalizes tensors with `mean 0.5` and `standard deviation 0.5`.

### Training Data

1. Loads and prepares the **MNIST** training dataset. 
2. Uses DataLoader for batching with shuffling.

### Testing Data

1. Loads and prepares the **MNIST** test dataset.
2. Uses DataLoader for batching without shuffling.

## Define the Neural Network Model & Set Up The Device

1. A simple feedforward neural network is defined with two fully connected layers.
2. `ReLU` activation is used after the first layer. 
3. The model is set to run on `GPU` if available, otherwise on `CPU`.

## Loss Function & Optimizer

1. Loss Function: Cross Entropy Loss to measure classification error.
2. Optimizer: Adam Optimizer to minimize loss with adaptive learning rates.

## Train The Model

1. Train the model over multiple epochs, here I used 10 epochs.
2. Computing loss and accuracy for each epoch.

## Evaluate the Model

1. Set the model to evaluation mode.
2. Compute test accuracy.
3. Visualize sample predictions.
4. Display a few test images with their actual and predicted labels.
