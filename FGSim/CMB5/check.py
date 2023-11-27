import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax = 1500
m = np.load('./40.npy')[0]

cl = hp.anafast(m, lmax=lmax, iter=100)
alm,_,n_iter = hp.map2alm_lsq(m, lmax=lmax, mmax=lmax, tol=1e-15, maxiter=20)
print(f'{n_iter}')
cl_lsq = hp.alm2cl(alm)

plt.semilogy(cl)
plt.semilogy(cl_lsq)
plt.show()
