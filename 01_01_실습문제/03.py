import torch

x = torch.tensor([1.0, 2.0, 3.0])

w = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)

b = torch.tensor(0.5, requires_grad=True)

y = x @ w + b  # Linear transformation

print("Output:", y.item())

y.backward()  # Compute gradients

print("Gradient w.r.t w:", w.grad)
print("Gradient w.r.t b:", b.grad)