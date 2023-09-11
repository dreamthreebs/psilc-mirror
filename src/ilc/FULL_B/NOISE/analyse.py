import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

cl_sum = 0
for i in range(100):
    print(f'loop:{i}')
    cl = np.load(f'./nilc_noise_cl2{i}.npy')
    cl_sum = cl_sum + cl

cl_avg = cl_sum/100
np.save('./nilc_noise_cl2avg.npy',cl_avg)

