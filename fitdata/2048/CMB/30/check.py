import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('./0.npy')
m1 = np.load('./99.npy')

# hp.mollview(m[0])
# hp.mollview(m[1])
# hp.mollview(m[2])
# plt.show()

lmax = 700
l = np.arange(lmax+1)
bl = hp.gauss_beam(fwhm=np.deg2rad(67)/60, lmax=lmax)

cl = hp.anafast(m, lmax=lmax)
cl1 = hp.anafast(m1, lmax=lmax)

plt.plot(l*(l+1)*cl[0]/(2*np.pi)/bl**2, label='cl_TT')
plt.plot(l*(l+1)*cl1[0]/(2*np.pi)/bl**2, label='cl_TT 1')
plt.show()


