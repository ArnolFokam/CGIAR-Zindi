import torch

# Define the transformation functions
def log_transform(x):
    return torch.log(x + 1.e-20)

def log_inverse(x):
    return torch.exp(x) - 1.e-20

def sqrt_transform(x):
    return torch.sqrt(x)

def sqrt_inverse(x):
    return x ** 2

# Create a dictionary of transformations
transformations = {
    "log": (log_transform, log_inverse),
    "sqrt": (sqrt_transform, sqrt_inverse),
}
