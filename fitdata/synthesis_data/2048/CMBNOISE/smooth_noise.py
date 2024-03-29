import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nside = 2048
beam_in = 17
beam_out = 30
lmax = 2000
m = np.load('../../../2048/NOISE/155/0.npy')[0]

bl_in = hp.gauss_beam(fwhm=np.deg2rad(beam_in)/60, lmax=lmax)
bl_out = hp.gauss_beam(fwhm=np.deg2rad(beam_out)/60, lmax=lmax)

almT_in = hp.map2alm(m, lmax=lmax)
almT_out = hp.almxfl(almT_in, bl_out/bl_in)

m_out = hp.alm2map(almT_out, lmax=lmax, nside=nside)

np.save('./155_to_95/noise_0.npy', m_out)
std = np.std(m_out)
print(f'{std=}')

hp.mollview(m_out, title='out')
hp.mollview(m, title='in')
# m_ref = np.load('./95/0.npy')[0]
# hp.mollview(m_ref, title='ref')

plt.show()


