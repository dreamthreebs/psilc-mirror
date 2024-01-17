import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

beam = 63
freq = 40
lmax = 500
l = np.arange(lmax+1)

m = np.load('./40GHz/0.npy')[0]
cl = hp.anafast(m, lmax=lmax)
bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)

cl_true = np.load('../../src/cmbsim/cmbdata/cl_realization/0.npy')[0, :lmax+1]

plt.plot(l*(l+1)*cl/(2*np.pi)/bl**2, label='cl')
plt.plot(l*(l+1)*cl_true/(2*np.pi), label='cl_true')
plt.legend()
plt.semilogy()
plt.show()

