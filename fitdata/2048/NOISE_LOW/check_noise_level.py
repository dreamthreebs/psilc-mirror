import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

noise_low = np.load('./270/0.npy')[2]
noise = np.load('../NOISE/270/0.npy')[2]

std_noise_low = np.std(noise_low)
std_noise = np.std(noise)
print(f'{std_noise_low=}')
print(f'{std_noise=}')


