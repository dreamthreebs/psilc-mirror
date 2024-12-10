import numpy as np
import timeit

# Sample data
D_ell_array = np.random.rand(1000000)  # Large array with 1 million elements
ell_array = np.arange(len(D_ell_array))

# Method with for loop
def compute_with_loop():
    C_ell_array = np.zeros_like(D_ell_array, dtype=np.float64)
    for i, ell in enumerate(ell_array):
        if ell > 1:
            C_ell_array[i] = (2 * np.pi * D_ell_array[i]) / (ell * (ell + 1))
        else:
            C_ell_array[i] = 0
    return C_ell_array

# Vectorized method
def compute_vectorized(D_ell):
    mask = ell_array > 1
    C_ell = np.zeros_like(D_ell, dtype=np.float64)
    C_ell[mask] = (2 * np.pi * D_ell[mask]) / (ell_array[mask] * (ell_array[mask] + 1))
    C_ell[~mask] = 0
    return C_ell

# Timing
loop_time = timeit.timeit(compute_with_loop, number=1)
vectorized_time = timeit.timeit(compute_vectorized, number=1)

print(f"Loop time: {loop_time:.4f} seconds")
print(f"Vectorized time: {vectorized_time:.4f} seconds")

