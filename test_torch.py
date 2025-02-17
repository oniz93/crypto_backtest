"""
test_torch.py
-------------
This script tests whether PyTorch can detect the MPS (Apple Silicon) backend.
It prints a tensor if MPS is available, otherwise it prints a message.
"""

import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print("MPS device not found.")
