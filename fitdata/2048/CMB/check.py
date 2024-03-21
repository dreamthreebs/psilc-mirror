import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('./155/0.npy')
m1 = np.load('./155_bak/0.npy')

l = np.arange(2001)
cl = hp.anafast(m, lmax=2000)[0]
cl1 = hp.anafast(m1, lmax=2000)[0]

plt.semilogy(l*(l+1)*cl/(2*np.pi), label='cl')
plt.semilogy(l*(l+1)*cl1/(2*np.pi), label='cl1')
plt.show()

