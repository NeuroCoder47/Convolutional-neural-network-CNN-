# Convolutional Neural Network (CNN) for Food Classification

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify images of food into three categories: Pizza, Steak, and Sushi. The model is trained on a dataset of food images and is evaluated on a test set to measure its performance.

## Dataset

The dataset used consists of images categorized into three classes:
- Pizza
- Steak
- Sushi

The dataset is divided into:
- Training set: `food/train/`
- Test set: `food/test/`

Each folder contains images for the respective food categories.

### Dataset Structure

- Training Data: 154 images of Pizza, 146 images of Steak, and 150 images of Sushi.
- Test Data: 46 images of Pizza, 58 images of Steak, and 46 images of Sushi.

## Model Architecture

The CNN architecture consists of the following components:
1. **Convolutional layers**: Extract features from the input images.
2. **Pooling layers**: Downsample the extracted features to reduce dimensionality.
3. **Fully connected layers**: Perform classification based on the extracted features.

The model is trained using the following:
- **Optimizer**: Adam
- **Loss function**: CrossEntropyLoss
- **Batch size**: 50
- **Number of epochs**: 10 (modifiable)
- **Device**: GPU (if available), otherwise CPU.

## Data Preprocessing

Before feeding the images to the model, the following transformations are applied:
- Resize the images to a fixed size of 224x224 pixels.
- Normalize the images using standard mean and standard deviation values.

## Training

The training process involves loading the data using PyTorch's `DataLoader`, iterating through the training set, and updating the model's weights using the backpropagation algorithm.

- **Train DataLoader**: 9 batches of 50 images each.
- **Test DataLoader**: 3 batches of 50 images each.

## Accuracy Function

An accuracy function is implemented to calculate the percentage of correctly predicted labels over the total number of samples.

```python
def accuracy_fn(y_true, y_pred):
    if len(y_pred.shape) == 1:
        _, predicted = torch.max(y_pred, dim=0)
    else:
        _, predicted = torch.max(y_pred, dim=1)
    correct = (predicted == y_true).sum().item()
    accuracy = correct / y_true.shape[0]
    return accuracy
