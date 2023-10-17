import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax=800
nside=512
l = np.arange(lmax+1)
mask = np.load('../../mask/north/BINMASKG.npy')
fsky = np.sum(mask)/np.size(mask)
m = np.load('./cmbtqunoB.npy')[0]

theory_cl = np.load('./cmbcl.npy')
full_cl = hp.anafast(m, lmax=lmax)

opt1 = hp.alm2map(hp.map2alm(m * mask, lmax=lmax), nside=nside)
cl1 = hp.anafast(opt1, lmax=lmax)
opt2 = hp.alm2map(hp.map2alm(opt1 * mask, lmax=lmax), nside=nside)
cl2 = hp.anafast(opt2, lmax=lmax)
opt3 = hp.alm2map(hp.map2alm(opt2 * mask, lmax=lmax), nside=nside)
cl3 = hp.anafast(opt3, lmax=lmax)





# plt.loglog(l*(l+1)*theory_cl[:lmax+1,0]/(2*np.pi) )
plt.loglog(l*(l+1)*full_cl/(2*np.pi), label='full_cl' )
plt.loglog(l*(l+1)*cl1/(2*np.pi)/fsky, label='cl1' )
plt.loglog(l*(l+1)*cl2/(2*np.pi)/fsky, label='cl2' )
plt.loglog(l*(l+1)*cl3/(2*np.pi)/fsky, label='cl3' )
plt.legend()
plt.show()
