# Important Model Building Classes in PyTorch

## Two Approaches to Model Building

### 1. **Transfer Learning**
- **Description**: Using pre-trained models from PyTorch's ecosystem
- **Sources**:
  - `torchvision.models` (ResNet, VGG, AlexNet, etc.)
  - `torch.hub` (models from various repositories)

### 2. **Custom Model Building**
- **Description**: Building models from scratch for specific/novel problems
- **Approach**: Using layers and activation functions as building blocks
- **Components**:
  - Layers like `nn.Linear`
  - Activation functions like `nn.ReLU`

## Core Building Classes

### `torch.nn`
- Contains all building blocks for computational graphs
- Provides layers, loss functions, and other neural network components

### `torch.nn.Parameter`
- Defines what parameters our model should learn
- Tracks tensors that should be optimized during training
- Automatically handles gradient computation

### `torch.nn.Module` (Base Class)
- **Purpose**: Base class for all neural network models
- **Usage**: Must subclass this class to create custom models
- **Key Method to Override**: `forward()` function

#### Example Structure:
```python
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers here
        self.layer1 = nn.Linear(10, 5)
        
    def forward(self, x):
        # Define forward computation
        return self.layer1(x)
```

### `def forward()`
- **Purpose**: Defines what happens in the forward computation
- **Note**: Never call this method directly; use `model(input_data)` instead
- **Automatic Handling**: Backward pass is automatically implemented via autograd

### `torch.optim`
- **Purpose**: Where optimization algorithms live
- **Responsibility**: Handles gradient descent and parameter updates
- **Common Optimizers**:
  - `optim.SGD` (Stochastic Gradient Descent)
  - `optim.Adam` (Adaptive Moment Estimation)
  - `optim.RMSprop`

## Typical Workflow
1. **Define model** by subclassing `nn.Module`
2. **Initialize optimizer** from `torch.optim`
3. **Implement forward pass** in `forward()` method
4. **Compute loss** and call `backward()` for gradients
5. **Update parameters** using optimizer's `step()` method
