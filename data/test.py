import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax = 300
l = np.arange(lmax+1)
bl = hp.gauss_beam(fwhm=np.deg2rad(63/60), lmax=lmax)
m1 = np.load('./band5pswym2/smcmb/data.npy')[4]
m2 = np.load('./band5pswym3/smcmb/data.npy')[4]
print(f'{m1.shape}')
cl1 = hp.anafast(m1, lmax=lmax)
cl2 = hp.anafast(m2, lmax=lmax)

plt.semilogy(l*(l+1)*cl1/(2*np.pi)/bl**2, label='lmax=300' )
plt.semilogy(l*(l+1)*cl2/(2*np.pi)/bl**2, label='lmax=500' )
plt.legend()
plt.show()
