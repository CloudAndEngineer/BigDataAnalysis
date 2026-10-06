import torch

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)

print(torch.relu(x))  # Apply ReLU activation function

for element in x:
    print(f"ReLU({element.item()}) = {torch.relu(element).item()}")  # Print ReLU for each element