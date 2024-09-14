import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax = 600
l = np.arange(lmax+1)
apo_mask = np.load('../../src/mask/north/APOMASKC1_5.npy')

m0 = np.load('./data/c/no_leakage.npy')
m1 = np.load('./data/c_check/eblc_226.npy')
m2 = np.load('./data/c_check/eblc_246.npy')
m3 = np.load('./data/c_check/eblc_266.npy')
m4 = np.load('./data/c_check/eblc_286.npy')
m5 = np.load('./data/c_check/eblc_306.npy')
m6 = np.load('./data/c_check/eblc_cmb.npy')

cl0 = hp.anafast(m0, lmax=lmax)
cl1 = hp.anafast(m1, lmax=lmax)
cl2 = hp.anafast(m2, lmax=lmax)
cl3 = hp.anafast(m3, lmax=lmax)
cl4 = hp.anafast(m4, lmax=lmax)
cl5 = hp.anafast(m5, lmax=lmax)
cl6 = hp.anafast(m6, lmax=lmax)


plt.figure(1)
plt.loglog(l**2*cl0, label='no leakage')
plt.loglog(l**2*cl6, label='2.06')
plt.loglog(l**2*cl1, label='2.26')
plt.loglog(l**2*cl2, label='2.46')
plt.loglog(l**2*cl3, label='2.66')
plt.loglog(l**2*cl4, label='2.86')
plt.loglog(l**2*cl5, label='3.06')

plt.legend()
plt.show()

# plt.figure(2)
# plt.plot((cl1-cl0)/cl0)
# plt.plot((cl2-cl0)/cl0)
# plt.show()

