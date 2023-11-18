import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nside = 64
lmax = 64

m = np.random.normal(loc=0, scale=1, size=(hp.nside2npix(nside)))
hp.mollview(m)
plt.show()

bl2 = hp.gauss_beam(fwhm=np.deg2rad(2), lmax=lmax)
bl1 = hp.gauss_beam(fwhm=np.deg2rad(1), lmax=lmax)
bl3 = hp.gauss_beam(fwhm=np.deg2rad(3), lmax=lmax)

m2 = hp.smoothing(m, fwhm=np.deg2rad(2))
m2_1 = hp.alm2map(hp.almxfl(hp.map2alm(m2, lmax=lmax), bl1/bl2), nside=nside)
m2_3 = hp.alm2map(hp.almxfl(hp.map2alm(m2, lmax=lmax), bl3/bl2), nside=nside)
m3 = hp.smoothing(m, fwhm=np.deg2rad(3))
m1 = hp.smoothing(m, fwhm=np.deg2rad(1))


l = np.arange(lmax+1)
cl = hp.anafast(m, lmax=lmax)
cl2 = hp.anafast(m2, lmax=lmax)
cl3 = hp.anafast(m3, lmax=lmax)
cl2_1 = hp.anafast(m2_1, lmax=lmax)
cl2_3 = hp.anafast(m2_3, lmax=lmax)
cl1 = hp.anafast(m1, lmax=lmax)

plt.semilogy(l*(l+1)*cl/(2*np.pi), label='cl')
plt.semilogy(l*(l+1)*cl2/(2*np.pi), label='cl2')
plt.semilogy(l*(l+1)*cl2_1/(2*np.pi), label='cl2_1')
plt.semilogy(l*(l+1)*cl2_3/(2*np.pi), label='cl2_3')
plt.semilogy(l*(l+1)*cl1/(2*np.pi), label='cl1')
plt.semilogy(l*(l+1)*cl3/(2*np.pi), label='cl3')
plt.legend()
plt.show()

