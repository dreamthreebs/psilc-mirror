import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

beam = 9
lmax = 1999
l = np.arange(lmax+1)

cl = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')
print(f'{cl.shape=}')

bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)

plt.loglog(l*(l+1)*cl[:,0]*bl**2/(2*np.pi), label='fwhm=9 arcmin')
plt.loglog(l*(l+1)*cl[:,0]/(2*np.pi), label='no beam')
plt.legend()
plt.show()

