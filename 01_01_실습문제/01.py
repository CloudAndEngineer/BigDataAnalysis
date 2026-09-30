import torch
import sys
import numpy as np

# python version check
print("Python version:", sys.version)

# torch version check
print("Torch version:", torch.__version__)

# numpy version check
print("Numpy version:", np.__version__)

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

# Check the device being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)