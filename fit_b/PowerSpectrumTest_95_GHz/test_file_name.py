import os

filename = "file1.1.npy"

try:
    with open(filename, "w") as f:
        pass  # Just test creating the file
    print(f"Filename '{filename}' is valid.")
except OSError as e:
    print(f"Filename '{filename}' is not valid: {e}")

