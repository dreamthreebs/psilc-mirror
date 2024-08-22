import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

beam = 20
l = np.arange(7000+1)
bl_temp = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,0]
bl_grad = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,1]
bl_curl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,2]

print(f'{bl_temp[0:50]=}')
print(f'{bl_grad[0:50]=}')
print(f'{bl_curl[0:50]=}')

print(f'{bl_temp[2000:2050]=}')
print(f'{bl_grad[2000:2050]=}')
# print(f'{bl_curl[0:50]=}')


# plt.semilogy(bl_temp, label='bl_temp')
# plt.semilogy(bl_grad, label='bl_grad')
# plt.semilogy(bl_curl, label='bl_curl')
# plt.plot()
# plt.legend()
# plt.show()
