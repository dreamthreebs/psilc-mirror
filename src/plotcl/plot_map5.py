import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax=350
l = np.arange(lmax+1)

mpilc = np.load('../../data/Rtol1k/simpilc/pilc_map.npy')
mhilc = np.load('../../data/Rtol1k/simhilc/hilc_map.npy')
mnilc = np.load('../../data/Rtol1k/simnilc/nilc_map0.npy')

mnspilc = np.load('../../data/Rtol1k/simpilc/pilc_noise_res_map.npy')
mnshilc = np.load('../../data/Rtol1k/simhilc/hilc_noise_res_map.npy')
mnsnilc = np.load('../../data/Rtol1k/simnilc/nilc_noise_res_map0.npy')


cmb = np.load('../../FGSim/CMB/270.npy')
cmb_b = hp.alm2map(hp.map2alm(cmb, lmax=lmax)[2], nside=512)

mask = np.load('../mask/north/APOMASKC1_10.npy')

clpilc = hp.anafast(mpilc, lmax=lmax)
clhilc = hp.anafast(mhilc, lmax=lmax)
clnilc = hp.anafast(mnilc, lmax=lmax)

clnspilc = hp.anafast(mnspilc, lmax=lmax)
clnshilc = hp.anafast(mnshilc, lmax=lmax)
clnsnilc = hp.anafast(mnsnilc, lmax=lmax)


clcmb = hp.anafast(cmb_b * mask, lmax=lmax)

plt.plot(l*(l+1)*clcmb/(2*np.pi), label='cmb')
plt.plot(l*(l+1)*(clpilc-clnspilc)/(2*np.pi), label='pilc')
plt.plot(l*(l+1)*(clhilc-clnshilc)/(2*np.pi), label='hilc')
plt.plot(l*(l+1)*(clnilc-clnsnilc)/(2*np.pi), label='nilc')

plt.legend()
plt.semilogy()
plt.show()








