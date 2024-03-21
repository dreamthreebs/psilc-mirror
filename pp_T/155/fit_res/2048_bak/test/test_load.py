import numpy as np
import healpy as hp
import time

# m = np.random.random(12* 16**2)

# np.save('test_map.npy', m)
# hp.write_map('test_map.fits', m, overwrite=True)

time0 = time.perf_counter()

m = np.load('./test_map.npy')
# m = hp.read_map('./test_map.fits', field=0)
print(f'{time.perf_counter()-time0}')

# import numpy as np
# import h5py

# # Generate synthetic data
# data = np.random.rand(100000, 10)  # 100,000 rows, 10 columns

# np.savetxt('data.csv', data, delimiter=',')
# # Save data as NPY
# np.save('data.npy', data)

# # Save data as HDF5
# with h5py.File('data.h5', 'w') as hf:
#     hf.create_dataset('data', data=data)


# import timeit

# # Benchmark CSV loading with numpy
# csv_time = timeit.timeit("np.loadtxt('data.csv', delimiter=',')", setup="import numpy as np", number=10) / 10
# print(f"Average loading time for CSV: {csv_time} seconds")

# # Benchmark NPY loading with numpy
# npy_time = timeit.timeit("np.load('data.npy')", setup="import numpy as np", number=10) / 10
# print(f"Average loading time for NPY: {npy_time} seconds")

# # Benchmark HDF5 loading with h5py
# hdf_time = timeit.timeit("with h5py.File('data.h5', 'r') as hf: hf['data'][:]", setup="import h5py", number=10) / 10
# print(f"Average loading time for HDF5: {hdf_time} seconds")

#import os
#import time
#import numpy as np
#
## Parameters
#file_path = 'test_file.bin'
#file_size_mb = 100  # Size of the file to write, in megabytes
#block_size = 4096  # Block size to use in bytes
#
#def write_test(file_path, file_size_mb, block_size):
#    num_blocks = (file_size_mb * 1024 * 1024) // block_size
#    data = np.random.bytes(block_size)
#
#    start_time = time.time()
#    with open(file_path, 'wb') as f:
#        for _ in range(num_blocks):
#            f.write(data)
#    end_time = time.time()
#
#    os.sync()  # Ensure all data is written to disk
#    duration = end_time - start_time
#    write_speed = file_size_mb / duration
#    return write_speed
#
#def read_test(file_path, block_size):
#    start_time = time.time()
#    with open(file_path, 'rb') as f:
#        while f.read(block_size):
#            pass
#    end_time = time.time()
#
#    duration = end_time - start_time
#    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
#    read_speed = file_size_mb / duration
#    return read_speed
#
#write_speed = write_test(file_path, file_size_mb, block_size)
#print(f"Write speed: {write_speed:.2f} MB/s")
#
#read_speed = read_test(file_path, block_size)
#print(f"Read speed: {read_speed:.2f} MB/s")
#
## Cleanup
#os.remove(file_path)
#
#
#
