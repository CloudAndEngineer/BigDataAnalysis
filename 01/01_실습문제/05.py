import torch
import matplotlib.pyplot as plt

x = torch.linspace(-3, 3, 100)  # Create a tensor of 100 points between -3 and 3
w = 1.0
b = 0.5

y_linear = w * x + b  # Linear function
y_relu = torch.relu(y_linear)  # Apply ReLU activation function

# plot y_linear and y_relu and add legend
plt.figure(figsize=(10, 6)) # Set the figure size for the plot
plt.plot(x.numpy(), y_linear.numpy(), label='Linear Function', color='blue')
plt.plot(x.numpy(), y_relu.numpy(), label='ReLU Activation', color='red')
# plot(): Shows the linear function and ReLU activation function on the same graph with labels and colors.
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Function and ReLU Activation')
plt.legend() # Add legend to the plot
plt.grid(True) # Add vertical and horizontal grid to the plot
plt.show() # Display the plot