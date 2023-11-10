import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m1 = np.load('./40.npy')
m2 = np.load('../../FGSim/FG_noPS5/40.npy')

hp.mollview(m1[2], title='wym')
hp.mollview(m2[2], title='zzr')
hp.mollview(m1[2]-m2[2], title='difference')
plt.show()

# lmax = 1500
# l = np.arange(lmax+1)
# cl1 = hp.anafast(m1[1], lmax=lmax)
# cl2 = hp.anafast(m2[1], lmax=lmax)
# cl_diff = hp.anafast(m1[1]-m2[1], lmax=lmax)

# plt.semilogy(l*(l+1)*cl1/(2*np.pi))
# plt.semilogy(l*(l+1)*cl2/(2*np.pi))

# plt.semilogy(l*(l+1)*cl_diff/(2*np.pi))
# plt.show()
