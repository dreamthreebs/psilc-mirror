import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('./cmbtqunoB20483.npy')
lmax=1000
l = np.arange(lmax+1)
nside=2048

# cl = hp.anafast(m, lmax=lmax)
# plt.loglog(l*(l+1)*cl[1]/(2*np.pi))
# plt.show()

sm1 = hp.smoothing(m, fwhm=np.deg2rad(1))

cl1 = hp.anafast(sm1, lmax=lmax)
plt.loglog(l*(l+1)*cl1[2]/(2*np.pi))
# plt.show()
bl = hp.gauss_beam(fwhm=np.deg2rad(1), pol=True, lmax=lmax)

alms = hp.map2alm(m, lmax=lmax)
almT, almE, almB = [alm for alm in alms]

almT = hp.almxfl(almT, bl[:,0])
almE = hp.almxfl(almE, bl[:,1])
almB = hp.almxfl(almB, bl[:,2])

sm2 = hp.alm2map([almT, almE, almB], nside=nside)
cl2 = hp.anafast(sm2, lmax=lmax)


# i = hp.smoothing(m[0], fwhm=1)
# q = hp.smoothing(m[1], fwhm=1)
# u = hp.smoothing(m[2], fwhm=1)

# cl2 = hp.anafast([i,q,u], lmax=lmax)
plt.loglog(l*(l+1)*cl2[2]/(2*np.pi))
plt.loglog(l*(l+1)*np.abs(cl2[2]-cl1[2])/(2*np.pi))
plt.show()
