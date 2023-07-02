# Check that MPS is available
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    device = "mps"

print(f"Using device: {device}", torch.device(device=device))
