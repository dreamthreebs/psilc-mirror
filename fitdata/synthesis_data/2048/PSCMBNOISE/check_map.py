import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

beam = 11
lmax = 2000
l = np.arange(lmax+1)
bl = hp.gauss_beam(fwhm=np.deg2rad(9)/60, lmax=lmax)

m = np.load('../../../2048/CMB/215/0.npy')
cl = hp.anafast(m, lmax=lmax)

plt.semilogy(l*(l+1)*cl[0]/(2*np.pi)/bl**2, label='TT')
plt.semilogy(l*(l+1)*cl[1]/(2*np.pi)/bl**2, label='EE')
plt.semilogy(l*(l+1)*cl[2]/(2*np.pi)/bl**2, label='BB')
plt.legend()
plt.show()

