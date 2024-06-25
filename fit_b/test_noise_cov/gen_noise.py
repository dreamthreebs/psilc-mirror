import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nside = 16
npix = hp.nside2npix(nside)
print(f'{npix=}')
nstd = 1

noise_list = []
noise_lmax10_list = []
noise_lmax47_list = []
for rlz_idx in range(npix+100):
    noise = nstd * np.random.normal(loc=0, scale=1, size=npix)
    noise_lmax47 = hp.alm2map(hp.map2alm(noise), nside=nside)
    noise_lmax10 = hp.alm2map(hp.map2alm(noise, lmax=10), nside=nside)
    
    noise_list.append(noise)
    noise_lmax10_list.append(noise_lmax10)
    noise_lmax47_list.append(noise_lmax47)

noise_arr = np.array(noise_list)
noise_lmax10_arr = np.array(noise_lmax10_list)
noise_lmax47_arr = np.array(noise_lmax47_list)
print(f'{noise_arr.shape=}')

noise_cov = np.cov(noise_arr, rowvar=False)
noise_10_cov = np.cov(noise_lmax10_arr, rowvar=False)
noise_47_cov = np.cov(noise_lmax47_arr, rowvar=False)
print(f'{noise_cov=}')
print(f'{noise_10_cov=}')
print(f'{noise_47_cov=}')


