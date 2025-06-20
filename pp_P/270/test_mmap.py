import numpy as np
import time

time0 = time.perf_counter()

m = np.load('../../fitdata/2048/CMB/155/1.npy')

print(f'{time.perf_counter() - time0}')

time1 = time.perf_counter()

m = np.load('../../fitdata/2048/CMB/155/2.npy', mmap_mode='r')
print(f'{time.perf_counter() - time0}')
