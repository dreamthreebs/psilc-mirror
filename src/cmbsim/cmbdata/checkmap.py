import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax = 1000
nside = 2048
beam = 63
l = np.arange(lmax+1)

m0 = np.load('./cmbmap.npy')[0]
m1 = np.load('./cmbmap1.npy')[0]
mask = np.zeros_like(m0)
cl = np.load('./cmbcl.npy')[:lmax+1,0]
print(f'{cl[0:10]=}')

cl0 = hp.anafast(m0, lmax=lmax)
cl1 = hp.anafast(m1, lmax=lmax)

ctr_ipix = hp.ang2pix(nside=nside, theta=0, phi=0, lonlat=True)
ctr_vec = hp.pix2vec(nside=nside, ipix=ctr_ipix)
ipix_disc = hp.query_disc(nside=nside, vec=ctr_vec, radius=np.deg2rad(63)/60 )
mask[ipix_disc] = 1

fsky = np.sum(mask) / np.size(mask)
print(f'{fsky=}')
# hp.mollview(mask * m0)
# plt.show()

cl_mask0 = hp.anafast(mask * m0, lmax=lmax)
cl_mask1 = hp.anafast(mask * m1, lmax=lmax)

plt.plot(l*(l+1)*cl0/(2*np.pi), label='cl0')
plt.plot(l*(l+1)*cl1/(2*np.pi), label='cl1')
plt.plot(l*(l+1)*cl/(2*np.pi), label='cl')
plt.plot(l*(l+1)*cl_mask0/(2*np.pi)/fsky, label='cl mask0')
plt.plot(l*(l+1)*cl_mask1/(2*np.pi)/fsky, label='cl mask1')
plt.legend()
plt.show()

# hp.mollview(m[0])
# plt.show()
