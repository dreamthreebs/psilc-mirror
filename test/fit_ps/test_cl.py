import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax = 1000
bl = hp.gauss_beam(fwhm=np.deg2rad(63)/60, lmax=lmax)
cl = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[:lmax+1,2]
cl = cl*bl**2
l = np.arange(lmax+1)

plt.semilogy((2*l+1)*cl[0:lmax+1])
plt.show()
