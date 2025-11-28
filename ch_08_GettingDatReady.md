# PyTorch Workflow: Getting Data Ready

This covers the first step of the PyTorch workflow: preparing data for machine learning models.

## Overview of PyTorch Workflow

The PyTorch workflow consists of six main steps:
1. Get data ready
2. Build a model
3. Fit the model to data (training)
4. Make predictions and evaluate the model (inference)
5. Save and load the model
6. Put it all together

## Step 1: Getting Data Ready

We create our own "dataset" by defining the rule for a straight line (y = weight * X + bias) and using it to generate our data.

In simple terms: we're playing the role of "nature" or "reality" by creating a hidden rule that our model will need to discover.

## Understanding Tensor Operations: Squeeze and Unsqueeze

Before diving into data preparation, let's understand the fundamental tensor operations of squeezing and unsqueezing. These are shape management tools for tensors.

Think of it like using List vs. T:
- `unsqueeze` is like taking a single item T and putting it into a list to get List[T]
- `squeeze` is like taking a List with only one item and getting that single item T back out

### Unsqueeze Example

```python
import torch

# Start with a 1D tensor (vector)
x = torch.tensor([1, 2, 3, 4])
print(x.shape)  # torch.Size([4])

# Add a new dimension at dimension 0 -> becomes shape [1, 4] (1 row, 4 columns)
x_unsqueezed_0 = x.unsqueeze(dim=0)
print(x_unsqueezed_0, x_unsqueezed_0.shape)
# Output: tensor([[1, 2, 3, 4]]), torch.Size([1, 4])

# Add a new dimension at dimension 1 -> becomes shape [4, 1] (4 rows, 1 column)
x_unsqueezed_1 = x.unsqueeze(dim=1)
print(x_unsqueezed_1, x_unsqueezed_1.shape)
# Output: tensor([[1],
#                 [2],
#                 [3],
#                 [4]]), torch.Size([4, 1])
```

### Squeeze Example

```python
# Start with a 2D tensor that has a "useless" dimension of size 1
y = torch.tensor([[1], [2], [3], [4]])  # Shape: [4, 1]
print(y.shape)  # torch.Size([4, 1])

# Squeeze it: removes ALL dimensions of size 1
y_squeezed = y.squeeze()
print(y_squeezed, y_squeezed.shape)
# Output: tensor([1, 2, 3, 4]), torch.Size([4])

# Example with a specific dimension
z = torch.randn(3, 1, 5)  # Shape: [3, 1, 5]
z_squeezed_dim1 = z.squeeze(dim=1)  # Tries to remove dimension 1 (size=1, so it works)
print(z_squeezed_dim1.shape)  # torch.Size([3, 5])

# This would do nothing because dimension 0 has size 3, not 1
z_squeezed_dim0 = z.squeeze(dim=0)
print(z_squeezed_dim0.shape)  # torch.Size([3, 1, 5]) -> unchanged
```

## Creating Our Data

Now let's create our dataset using a linear relationship:

```python
import torch

# Define weight and bias
w = 0.1
b = 0.02

# Create input data
start = 1
end = 0
step = -0.02
x = torch.arange(start, end, step).unsqueeze(dim=1)
print(x)
```

Output:
```
tensor([[1.0000],
        [0.9800],
        [0.9600],
        [0.9400],
        [0.9200],
        [0.9000],
        [0.8800],
        [0.8600],
        [0.8400],
        [0.8200],
        [0.8000],
        [0.7800],
        [0.7600],
        [0.7400],
        [0.7200],
        [0.7000],
        [0.6800],
        [0.6600],
        [0.6400],
        [0.6200],
        [0.6000],
        [0.5800],
        [0.5600],
        [0.5400],
        [0.5200],
        [0.5000],
        [0.4800],
        [0.4600],
        [0.4400],
        [0.4200],
        [0.4000],
        [0.3800],
        [0.3600],
        [0.3400],
        [0.3200],
        [0.3000],
        [0.2800],
        [0.2600],
        [0.2400],
        [0.2200],
        [0.2000],
        [0.1800],
        [0.1600],
        [0.1400],
        [0.1200],
        [0.1000],
        [0.0800],
        [0.0600],
        [0.0400],
        [0.0200]])
```

## Creating Labeled Data

```python
# Create output/labeled data using the linear rule
y = w * x + b
print(y)
```

Output:
```
tensor([[0.1200],
        [0.1180],
        [0.1160],
        [0.1140],
        [0.1120],
        [0.1100],
        [0.1080],
        [0.1060],
        [0.1040],
        [0.1020],
        [0.1000],
        [0.0980],
        [0.0960],
        [0.0940],
        [0.0920],
        [0.0900],
        [0.0880],
        [0.0860],
        [0.0840],
        [0.0820],
        [0.0800],
        [0.0780],
        [0.0760],
        [0.0740],
        [0.0720],
        [0.0700],
        [0.0680],
        [0.0660],
        [0.0640],
        [0.0620],
        [0.0600],
        [0.0580],
        [0.0560],
        [0.0540],
        [0.0520],
        [0.0500],
        [0.0480],
        [0.0460],
        [0.0440],
        [0.0420],
        [0.0400],
        [0.0380],
        [0.0360],
        [0.0340],
        [0.0320],
        [0.0300],
        [0.0280],
        [0.0260],
        [0.0240],
        [0.0220]])
```

## Splitting Data into Training and Test Sets

Since we don't have separate training and test data, we split our available data:

```python
train_split = int(0.8 * len(x))  # Calculate the index for an 80/20 split
x_train, y_train = x[:train_split], y[:train_split]  # First 80% of data
x_test, y_test = x[train_split:], y[train_split:]    # Last 20% of data9
```

## Visualizing the Data

Instead of printing the data, let's visualize it:

```python
import torch
from torch import nn
import matplotlib.pyplot as plt

def plot_predictions(train_data=x_train,
                    train_labels=y_train,
                    test_data=x_test,
                    test_labels=y_test,
                    predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    plt.legend(prop={"size": 14})

plot_predictions()
```

This code generates a scatter plot showing:
- Blue points: Training data (first 80% of our dataset)
- Green points: Testing data (last 20% of our dataset)

The plot demonstrates a clear linear relationship between x and y values, which is exactly what we would expect given our data generation process (y = 0.1x + 0.02).
