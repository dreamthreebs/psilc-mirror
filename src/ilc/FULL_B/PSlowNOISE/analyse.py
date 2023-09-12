import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

num_sim = 50
cl_sum = 0

for i in range(num_sim):
    print(f'loop:{i}')
    cl = np.load(f'./nilc_noise_cl9sim{i}.npy')
    cl_sum = cl_sum + cl

cl_avg = cl_sum/num_sim
np.save('./nilc_noise_cl9avg.npy',cl_avg)

