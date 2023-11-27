import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('./40.npy')

lmax = 2000
l = np.arange(lmax+1)
cl = hp.anafast(m, lmax=lmax)
print(f'{cl.shape=}')

alm,_,niter = hp.map2alm_lsq(m, lmax=lmax, mmax=lmax, tol=1e-15, maxiter=30)
print(f'{niter=}')
cl_lsq = hp.alm2cl(alm)

for i, type_cl in enumerate("TEB"):
    plt.semilogy(l*(l+1)*cl[i]/(2*np.pi), label=f"{type_cl}")
    plt.semilogy(l*(l+1)*cl_lsq[i]/(2*np.pi), label=f"{type_cl} lsq")
    plt.show()
