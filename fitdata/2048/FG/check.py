import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax = 1999
l = np.arange(lmax + 1)
freq = 270

m = np.load(f'./{freq}/fg.npy')
m = np.load(f'../../synthesis_data/2048/PSCMBFGNOISE/270/1.npy')
hp.mollview(m[1],title='Q', min=-30, max=90)
hp.mollview(m[2],title='U', min=-44, max=53)
plt.show()

cl = hp.anafast(m, lmax=lmax)
plt.loglog(l, l*(l+1)*cl[2]/(2*np.pi))
plt.show()


