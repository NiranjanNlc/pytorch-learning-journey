# Neural Network Classification - Learning Notes

## 1. Classification Problems

### Definition
- **Classification**: Putting given things into categories based on previously labeled data
- **Regression**: Predicting data based on previous data sets

### Types of Classification
- **Binary Classification**: Only two options
  - Example: Spam vs Not Spam email
- **Multi-class Classification**: More than two categories
  - Example: Identifying different animals (dog, cat, etc.)

## 2. Neural Network Architecture for Classification

### Components
- **Input Layer**: Number of features the model expects
  - Determines how much data to process at start
- **Hidden Layers**: Extra layers between input and output
  - Find patterns and make combinations
  - More layers/steps = deeper learning
- **Neurons per Hidden Layer**: 
  - Like people examining same inputs in different ways
  - More neurons = better at spotting complex features
  - Too many can lead to slowdown
- **Output Layer**: Number of outputs for prediction
  - **Binary**: 1 output
  - **Multi-class**: 1 output per class

## 3. Activation Functions

### Hidden Layer Activation
- **Purpose**: Adds curves to handle non-linear problems
- Without this, networks struggle to separate groups

### Output Layer Activation
- **Binary Classification**: Sigmoid function (outputs 0 to 1)
- **Multi-class Classification**: Softmax function
  - Turns outputs into probabilities that add up to 1
  - Example: Softmax([2, 0.5, 0.1]) = [0.7, 0.2, 0.1]

## 4. Training Components

### Loss Function
- Measures how wrong the model is
- Guides model improvement during training

### Optimizer
- Determines how the network updates itself to improve
- Helps improve faster by analyzing mistakes

## Key Concepts
- **Hyperparameters**: Settings that define network architecture
- **Pattern Recognition**: Hidden layers look for patterns in data
- **Probability Output**: Classification outputs represent category probabilities
