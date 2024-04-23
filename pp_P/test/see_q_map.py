import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

freq = 155
lmax = 2000
l = np.arange(lmax+1)
dl_factor = (2*l+1) / (2*np.pi)
bl = hp.gauss_beam(fwhm=np.deg2rad(17)/60, lmax=lmax)
m_q = np.load(f'../../fitdata/2048/CMB/{freq}/77.npy').copy()

cl = hp.anafast(m_q, lmax=lmax)
plt.semilogy(cl[0]*dl_factor/bl**2)
plt.semilogy(cl[1]*dl_factor/bl**2)
plt.semilogy(cl[2]*dl_factor/bl**2)
plt.show()

