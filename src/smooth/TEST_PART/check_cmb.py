import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m1 = np.load('./noPS_northNOI/CMB/270.npy')
m2 = np.load('./noPS_northNOISM1/CMB/30.npy')

mask = np.load('../../mask/north/BINMASKG.npy')
apo_mask = np.load('../../mask/north_smooth/APOMASKC1_10.npy')

hp.orthview(m1[0]*mask, rot=[100,50,0], half_sky=True)
hp.orthview(m2[0], rot=[100,50,0], half_sky=True)
hp.orthview((m1[0]*mask-m2[0]), rot=[100,50,0], half_sky=True)
plt.show()

# hp.mollview(m1[0]-m2[0], rot=[100,50,0], half_sky=True)
# hp.mollview(m1[0]-m2[0])

lmax=500
l = np.arange(lmax+1)
cl = hp.anafast((m1[2]*mask-m2[2]) * apo_mask, lmax=lmax)
plt.semilogy(l*(l+1)*cl/(2*np.pi))

plt.xlabel('$\\ell$')
plt.ylabel('$D_\\ell$')
plt.show()


