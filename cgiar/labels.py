import torch

# Define the transformation functions
def log_transform(x):
    return torch.log(x + 1e-20)

def log_inverse(x):
    return torch.exp(x) - 1e-20

def sqrt_transform(x):
    return torch.sqrt(x)

def sqrt_inverse(x):
    return x ** 2

def inverse_transform(x):
    return 1 / (x + 1e-20)

def inverse_inverse(x):
    return (1 / x) - 1e-20

def exponential_transform(x):
    return torch.exp(x)

def exponential_inverse(x):
    return torch.log(x)

def rank_transform(x):
    sorted_x, rank_indices = x.sort()
    rank_x = sorted_x.clone()
    rank_x[rank_indices] = torch.arange(1, len(x) + 1, dtype=x.dtype, device=x.device)
    return rank_x

def rank_inverse(x):
    return x

def power_transform(x, power):
    return x ** power

def power_inverse(x, power):
    return x ** (1 / power)

# Create a dictionary of transformations
transformations = {
    "log": (log_transform, log_inverse),
    "sqrt": (sqrt_transform, sqrt_inverse),
    "inv": (inverse_transform, inverse_inverse),
    "exp": (exponential_transform, exponential_inverse),
    "rank": (rank_transform, rank_inverse),
    "pow": (power_transform, power_inverse),
}
