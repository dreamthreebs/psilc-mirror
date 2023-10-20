import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax = 500
l = np.arange(lmax+1)
nside = 512

m = np.load('../../FGSim/CMB/270.npy')
true = hp.alm2map(hp.map2alm(m, lmax=lmax)[2], nside=nside) * np.load('../mask/north/APOMASKC1_10.npy')
true_cl = hp.anafast(true, lmax=lmax)

pilc_cl = np.load('./pilcres/pilc_cl.npy')
pilc_fgres_cl = np.load('./pilcres/pilc_fgres_cl.npy')

hilc_cl = np.load('./hilcres/hilc_cl.npy')
hilc_fgres_cl = np.load('./hilcres/hilc_fgres_cl.npy')

nilc_cl = np.load('./nilcres/nilc_cl0.npy')
nilc_fgres_cl = np.load('./nilcres/nilc_fgres_cl0.npy')

pilc_cl1 = np.load('./pilcres/pilc_cl1.npy')
pilc_fgres_cl1 = np.load('./pilcres/pilc_fgres_cl1.npy')

hilc_cl1 = np.load('./hilcres/hilc_cl1.npy')
hilc_fgres_cl1 = np.load('./hilcres/hilc_fgres_cl1.npy')

nilc_cl1 = np.load('./nilcres/nilc_cl1.npy')
nilc_fgres_cl1 = np.load('./nilcres/nilc_fgres_cl1.npy')


plt.semilogy(l*(l+1)*true_cl/(2*np.pi), label='true_cl', color='black')
# plt.semilogy(l*(l+1)*pilc_cl/(2*np.pi), label='pilc_cl')
plt.semilogy(l*(l+1)*pilc_fgres_cl/(2*np.pi), label='pilc_fgres_cl')
# plt.semilogy(l*(l+1)*hilc_cl/(2*np.pi), label='hilc_cl')
plt.semilogy(l*(l+1)*hilc_fgres_cl/(2*np.pi), label='hilc_fgres_cl')
# plt.semilogy(l*(l+1)*nilc_cl/(2*np.pi), label='nilc_cl')
plt.semilogy(l*(l+1)*nilc_fgres_cl/(2*np.pi), label='nilc_fgres_cl')

# plt.semilogy(l*(l+1)*pilc_cl1/(2*np.pi), label='pilc_cl1')
plt.semilogy(l*(l+1)*pilc_fgres_cl1/(2*np.pi), label='pilc_fgres_cl1')
# plt.semilogy(l*(l+1)*hilc_cl1/(2*np.pi), label='hilc_cl1')
plt.semilogy(l*(l+1)*hilc_fgres_cl1/(2*np.pi), label='hilc_fgres_cl1')
# plt.semilogy(l*(l+1)*nilc_cl1/(2*np.pi), label='nilc_cl1')
plt.semilogy(l*(l+1)*nilc_fgres_cl1/(2*np.pi), label='nilc_fgres_cl1')


# plt.loglog()

plt.legend()
plt.show()

