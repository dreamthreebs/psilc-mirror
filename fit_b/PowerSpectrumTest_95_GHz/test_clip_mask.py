import numpy as np
import time

# Create a large random array with negative and positive values
arr = np.random.uniform(-10, 10, size=10**7)

# Using np.clip
arr_copy = arr.copy()  # Ensure both tests use the same initial data
start = time.time()
arr_clipped = np.clip(arr_copy, 0, None)
end = time.time()
print(f"np.clip: {end - start:.6f} seconds")

# Using conditional masking
arr_copy = arr.copy()  # Reset array
start = time.time()
arr_copy[arr_copy < 0] = 0
end = time.time()
print(f"Conditional masking: {end - start:.6f} seconds")

