import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax = 2000
bl10 = hp.gauss_beam(fwhm=np.deg2rad(10)/60, lmax=lmax)
bl60 = hp.gauss_beam(fwhm=np.deg2rad(60)/60, lmax=lmax)
bl30 = hp.gauss_beam(fwhm=np.deg2rad(30)/60, lmax=lmax)

plt.semilogy(bl10, label='bl10')
plt.semilogy(bl30, label='bl30')
plt.semilogy(bl60, label='bl60')

plt.semilogy((bl10/bl60)**2, label='bl10/60 square')
plt.semilogy((bl30/bl60)**2, label='bl30/60 square')

plt.show()


