import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('./95.npy')

lmax = 2000
l = np.arange(lmax+1)
cl = hp.anafast(m, lmax=lmax)
print(f'{cl.shape=}')


for i, type_cl in enumerate("TEB"):
    plt.semilogy(l*(l+1)*cl[i]/(2*np.pi), label=f"{type_cl}")
    plt.show()
