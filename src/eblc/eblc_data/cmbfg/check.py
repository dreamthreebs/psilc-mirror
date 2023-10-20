import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('./270.npy')
m1 = np.load('../test/cutqufitqu/270.npy')
apo_mask = np.load('../../../mask/north/APOMASKC1_2.npy')

lmax=500
l = np.arange(lmax+1)


hp.orthview(m, rot=[100,50,0], half_sky=True)
# cl = hp.anafast((m-m1) * apo_mask, lmax=300)
# plt.semilogy(l*(l+1)*cl/(2*np.pi))
plt.show()
