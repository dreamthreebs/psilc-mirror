import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nside = 512
lmax = 3*nside - 1

npix = hp.nside2npix(nside)

nstd = 1
noise = nstd * np.random.normal(loc=0, scale=1, size=(3,npix))

ori_std = np.std(noise[1])
print(f'{ori_std=}')

alm_i, alm_e, alm_b = hp.map2alm(noise, lmax=lmax)

# m_e_fml = hp.alm2map([0, alm_e, 0], nside=nside)
m_b = hp.alm2map(alm_b, nside=nside)
m_b_fml = hp.alm2map([np.zeros_like(alm_b), np.zeros_like(alm_b), alm_b], nside=nside)
m_b_fml_b = hp.alm2map(hp.map2alm(m_b_fml, lmax=lmax)[2], nside=nside)
print(f'{m_b_fml.shape=}')

std_b = np.std(m_b)
std_b_fml = np.std(m_b_fml[2])
std_b_fml_b = np.std(m_b_fml_b)

print(f"{std_b=}")
print(f"{std_b_fml=}")
print(f"{std_b_fml_b=}")



