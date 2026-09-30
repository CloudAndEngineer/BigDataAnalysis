import numpy as np

x = np.array([1, 2, 3])
w = np.array([0.1, 0.2, 0.3])

b = 0.5

y = x @ w + b  # Using the @ operator for matrix multiplication
print("Result of the linear operation:", y)

# raise exception when the shapes of x and w are not compatible for matrix multiplication
try:
    x_invalid = np.array([1, 2])  # Invalid shape for multiplication with w
    y_invalid = x_invalid @ w + b
except ValueError as e:
    print("Error:", e)