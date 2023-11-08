import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


lmax=500
l = np.arange(lmax+1)
nside=512
m0 = np.load('../../data/band5ps/simnilc/nilc_noise_res_map0.npy')
cl0 = hp.anafast(m0, lmax=lmax)

cl_sum = 0
for i in range(10):
    m1 = np.load(f'../../data/band5ps/NOISENILC/nilc_noise_res_map{i}.npy')
    cl1 = hp.anafast(m1, lmax=lmax)
    cl_sum = cl_sum + cl1

cl = cl_sum / 10


plt.semilogy(l*(l+1)*cl0/(2*np.pi))
plt.semilogy(l*(l+1)*cl/(2*np.pi))
plt.show()

