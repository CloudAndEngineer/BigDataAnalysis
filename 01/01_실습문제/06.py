import torch

x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) # shape: (2, 3)

w1 = torch.tensor([[0.1, 0.2], [0.3, 0.2] , [0.4, 0.3]], requires_grad=True) # shape: (3, 2)

b1 = torch.tensor([0.1, 0.2], requires_grad=True) # shape: (2,)

# x @ w1 + b1 will have shape (2, 2) because x has shape (2, 3) and w1 has shape (3, 2)

w2 = torch.tensor([[0.1], [0.2]], requires_grad=True) # shape: (2, 1)

b2 = torch.tensor([0.1], requires_grad=True)

try:
    malformed_x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.relu(malformed_x @ w1 + b1) @ w2 + b2
except RuntimeError as e:
    print(e)

y = torch.relu(x @ w1 + b1) @ w2 + b2 # shape: (2, 1)

print(y)