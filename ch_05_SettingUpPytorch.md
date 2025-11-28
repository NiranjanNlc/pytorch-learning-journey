# My PyTorch Installation Experience

Since I am interested in beginner-only projects and small projects, I assume that CPU is just enough and no NVIDIA GPU is required to run deep learning models using Python.

So, I have installed PyTorch using pip and verified the installation without using CUDA.

## The Challenge I Ran Into

I tried to verify PyTorch by opening the Python interpreter from the terminal, but ultimately got an error. 

The issue I discovered was that PyTorch required Python version above 3.x, while my Python version was below that requirement.

## Solution

I quickly opened the Python3 interpreter by typing `python3` in the terminal and successfully verified the installation of PyTorch.

## Verification Code

The code I used to verify PyTorch was:

```python
import torch
x = torch.rand(5, 3)
print(x)
```

This successfully created a random tensor and confirmed that PyTorch was working correctly with my CPU-only installation.
