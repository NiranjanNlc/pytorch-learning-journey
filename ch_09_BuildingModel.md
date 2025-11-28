# Building a Linear Regression Model with PyTorch

## Introduction

We will focus on creating neural network models capable of learning patterns in our data. The data we generate follows specific rules, and our machine learning goal is to discover these rules using input-output pairs.

### Model Selection Strategy

Start with simplest reasonable model (like linear)
↓
Evaluate performance
↓
If performance unacceptable → Try more complex model
↓
If too complex/overfitting → Simplify or get more data
↓
If business needs explanations → Use interpretable model
↓
Choose best trade-off between accuracy, speed, and interpretability

Model selection isn't about finding the "best" algorithm - it's about finding the best trade-off for our specific business context.

## Linear Model

We'll start with a simple linear model:

```
y_pred = w * x + b
```

Where:
- w = weight (a number to be learned)
- b = bias (a number to be learned)
- x = input (known)
- y_pred = prediction (will compare to true y)

### Initial State

We start with random values for w and b:
- w = 0.8, b = -0.2 (terrible guesses!)
- Input: x = 0.5 → Prediction: y_pred = 0.8*0.5 - 0.2 = 0.20
- True answer: y = 0.07
- Error = 0.20 - 0.07 = 0.13 (this is bad!)

Our model's job is to slowly adjust w and b to minimize the error.

### Learning Process

The learning process works as follows:
1. Make a prediction
2. Measure how wrong it is (calculate error)
3. Adjust w and b slightly to reduce error
4. Repeat thousands of times

## Manual Implementation

Let's code manually to illustrate our approach:

```python
x = 0.5
y = 0.07

w = 0.7  # Should approach 0.1
b = 0.5  # Should approach 0.02

def prediction(x):
    return w * x + b

# Testing model
print("prediction is " + str(prediction(0.5)))
print("error is " + str(y - prediction(0.5)))
```

Output:
```
prediction is 0.85
error is -0.78
```

### Making the Model Learnable

```python
def error(y_true, y_pred):
    return y_true - y_pred

# Gradients calculation
# ∂error/∂w = ∂(w*x + b - y_true)/∂w = x
grad_w = x  # = 0.5

# ∂error/∂b = ∂(w*x + b - y_true)/∂b = 1
grad_b = 1  # = 1

learning_rate = 0.1
# How big of steps we take
w = w + grad_w * learning_rate
b = b + grad_b * learning_rate

print("w is " + str(w))
print("b is " + str(b))
print("prediction is " + str(prediction(0.5)))
print("error is " + str(error(y, prediction(0.5))))
```

Output:
```
w is 0.75
b is 0.6
prediction is 0.975
error is -0.905
```

## PyTorch Implementation

In real applications, we don't need to code everything manually. PyTorch handles the low-level details.

### Complete Learning Loop with PyTorch

```python
import torch

# Initialize parameters with requires_grad=True
w = torch.tensor([-0.8], requires_grad=True)  # Initial weight guess
b = torch.tensor([0.2], requires_grad=True)   # Initial bias guess

# Training data
x_data = torch.tensor([0.5, 0.8, 0.3, 0, 0.081])      # Input features
y_true_data = torch.tensor([0.07, 0.10, 0.05, 0.03, 0.031])  # Correct outputs

# Training loop - run for 100 epochs
for epoch in range(100):
    total_error = 0  # Track cumulative error for this epoch
    
    # Process each training example one by one
    for x, y_true in zip(x_data, y_true_data):
        # Forward pass
        y_pred = w * x + b
        
        # Calculate squared error
        error = (y_pred - y_true)**2
        
        # Backward pass (gradient calculation)
        error.backward()
        
        # Parameter update
        with torch.no_grad():
            # Gradient descent update
            w -= 0.1 * w.grad  # Adjust weight
            b -= 0.1 * b.grad  # Adjust bias
            
            # Reset gradients for next iteration
            w.grad.zero_()
            b.grad.zero_()
        
        # Accumulate error for monitoring
        total_error += error.item()
    
    # Monitoring progress
    if epoch % 10 == 0:
        avg_error = total_error / len(x_data)
        print(f"Epoch {epoch}:")
        print(f" Average Error: {avg_error:.6f}")
        print(f" Current w: {w.item():.4f}")
        print(f" Current b: {b.item():.4f}")

# After training, check final values
print("Training completed!")
print(f"Final w: {w.item():.4f} (should be close to 0.1)")
print(f"Final b: {b.item():.4f} (should be close to 0.02)")
```

### Training Progress

| Epoch | Average Error | Weight (w) | Bias (b) |
|-------|---------------|------------|----------|
| 0     | 0.069798      | -0.7069    | 0.2332   |
| 10    | 0.010995      | -0.2495    | 0.1239   |
| 20    | 0.002051      | -0.0561    | 0.0681   |
| 30    | 0.000388      | 0.0273     | 0.0440   |
| 40    | 0.000078      | 0.0633     | 0.0336   |
| 50    | 0.000020      | 0.0788     | 0.0291   |
| 60    | 0.000010      | 0.0855     | 0.0272   |
| 70    | 0.000008      | 0.0884     | 0.0264   |
| 80    | 0.000007      | 0.0896     | 0.0260   |
| 90    | 0.000007      | 0.0902     | 0.0258   |

**Training completed!**
Final w: 0.0904 (should be close to 0.1)
Final b: 0.0258 (should be close to 0.02)

## Structured PyTorch Implementation

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Fix data inconsistency
x_data = torch.tensor([0.5, 0.8, 0.3, 0.0, 0.081])  # 5 input features
y_true_data = torch.tensor([0.07, 0.10, 0.05, 0.03, 0.031])  # 5 target values

# Define a proper PyTorch model class
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define model parameters
        self.weight = nn.Parameter(torch.tensor([-0.8]))  # Initial weight
        self.bias = nn.Parameter(torch.tensor([0.2]))     # Initial bias
    
    def forward(self, x):
        # Define the computation: y = w*x + b
        return self.weight * x + self.bias

# Create model instance
model = LinearRegressionModel()

# Training parameters
learning_rate = 0.1
num_epochs = 100

# Track training progress
losses = []
weights = []
biases = []

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    
    # Process each training example
    for x, y_true in zip(x_data, y_true_data):
        # Forward pass
        y_pred = model(x)
        
        # Calculate loss
        loss = (y_pred - y_true)**2
        
        # Backward pass
        loss.backward()
        
        # Update parameters (using gradient descent)
        with torch.no_grad():
            model.weight -= learning_rate * model.weight.grad
            model.bias -= learning_rate * model.bias.grad
        
        # Reset gradients
        model.weight.grad.zero_()
        model.bias.grad.zero_()
        
        total_loss += loss.item()
    
    # Record progress
    avg_loss = total_loss / len(x_data)
    losses.append(avg_loss)
    weights.append(model.weight.item())
    biases.append(model.bias.item())
    
    # Print progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss={avg_loss:.6f}, w={model.weight.item():.4f}, b={model.bias.item():.4f}")

# Final evaluation
print("\nTraining completed!")
print(f"Final parameters: w={model.weight.item():.6f}, b={model.bias.item():.6f}")

# Plot training progress
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(losses)
plt.title('Training loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 3, 2)
plt.plot(weights)
plt.title('Weight Convergence')
plt.xlabel('Epoch')
plt.ylabel('Weight')

plt.subplot(1, 3, 3)
plt.plot(biases)
plt.title('Bias Convergence')
plt.xlabel('Epoch')
plt.ylabel('Bias')

plt.tight_layout()
plt.show()

# Test the model on new data
test_x = torch.tensor([0.25, 0.6, 0.9])
with torch.no_grad():  # No need to track gradients during inference
    test_predictions = model(test_x)

print("\nTest predictions:")
for x, pred in zip(test_x, test_predictions):
    print(f"x={x:.3f} -> y_pred={pred:.4f}")

# Compare with true relationship (y = 0.1*x + 0.02)
print("\nComparison with true relationship (y = 0.1*x + 0.02):")
for x, pred in zip(test_x, test_predictions):
    true_y = 0.1 * x + 0.02
    print(f"x={x:.3f}: Predicted={pred:.4f}, True={true_y:.4f}, Error={abs(pred - true_y):.4f}")
```

### Results

**Training Output:**
```
Epoch 0: Loss=0.083758, w=-0.7069, b=0.2332
Epoch 10: Loss=0.013193, w=-0.2495, b=0.1239
Epoch 20: Loss=0.002462, w=-0.0561, b=0.0681
Epoch 30: Loss=0.000465, w=0.0273, b=0.0440
Epoch 40: Loss=0.000094, w=0.0633, b=0.0336
Epoch 50: Loss=0.000025, w=0.0788, b=0.0291
Epoch 60: Loss=0.000012, w=0.0855, b=0.0272
Epoch 70: Loss=0.000009, w=0.0884, b=0.0264
Epoch 80: Loss=0.000009, w=0.0896, b=0.0260
Epoch 90: Loss=0.000009, w=0.0902, b=0.0258

Training completed!
Final parameters: w=0.090391, b=0.025782
```

**Test Predictions:**
```
Test predictions:
x=0.250 -> y_pred=0.0484
x=0.600 -> y_pred=0.0800
x=0.900 -> y_pred=0.1071

Comparison with true relationship (y = 0.1*x + 0.02):
x=0.250: Predicted=0.0484, True=0.0450, Error=0.0034
x=0.600: Predicted=0.0800, True=0.0800, Error=0.0000
x=0.900: Predicted=0.1071, True=0.1100, Error=0.0029
```

## Conclusion

We've successfully built and trained a linear regression model using PyTorch. The model learned to approximate the relationship y = 0.1*x + 0.02 from the training data. The final parameters (w=0.0904, b=0.0258) are close to the true values (w=0.1, b=0.02), demonstrating the effectiveness of gradient descent in learning linear relationships.

The PyTorch framework handled the low-level details of gradient calculation and parameter updates, allowing us to focus on the high-level model design.
