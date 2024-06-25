import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

noise = np.load('./data/noise.npy')
lmax = 1000
nside = 2048

alm_t, alm_e, alm_b = hp.map2alm(noise, lmax=lmax)
noise_e = hp.alm2map(alm_e, nside=nside)
noise_b = hp.alm2map(alm_b, nside=nside)

# hp.mollview(noise[1], title='Q')
# hp.mollview(noise[2], title='U')
# plt.show()

hp.mollview(noise_e, title='E')
hp.mollview(noise_b, title='B')
plt.show()

std_q = np.std(noise[1])
std_u = np.std(noise[2])
print(f'{std_q=}')
print(f'{std_u=}')

std_e = np.std(noise_e)
std_b = np.std(noise_b)
print(f'{std_e=}')
print(f'{std_b=}')


