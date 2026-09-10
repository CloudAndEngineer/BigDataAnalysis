import torch
import numpy as np

a = np.array([1, 2, 3]) # Output: [1 2 3]
x = torch.tensor([1, 2, 3]) # Output: tensor([1, 2, 3])

y = x * 2

print(a * 2)

print(y)

print(torch.__version__)
print(torch.cuda.is_available())