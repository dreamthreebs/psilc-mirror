import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax=300
nside=512
l = np.arange(lmax+1)
mask = np.load('../../src/mask/north/BINMASKG.npy')

mpilc = np.load('../../newdata/band5ps350/simpilc/pilc_map.npy')
mhilc = np.load('../../newdata/band5ps350/simhilc/hilc_map.npy')
mnilc = np.load('../../newdata/band5ps350/simnilc/nilc_map0.npy')

# mpilc = hp.alm2map(hp.map2alm(mpilc, lmax=lmax), nside=nside)
# mhilc = hp.alm2map(hp.map2alm(mhilc, lmax=lmax), nside=nside)
# mnilc = hp.alm2map(hp.map2alm(mnilc, lmax=lmax), nside=nside)

mpfgres = np.load('../../newdata/band5ps350/simpilc/pilc_fgres_map.npy')
mhfgres = np.load('../../newdata/band5ps350/simhilc/hilc_fgres_map.npy')
mnfgres = np.load('../../newdata/band5ps350/simnilc/nilc_fgres_map0.npy')

# mpfgres = hp.alm2map(hp.map2alm(mpfgres, lmax=lmax), nside=nside)
# mhfgres = hp.alm2map(hp.map2alm(mhfgres, lmax=lmax), nside=nside)
# mnfgres = hp.alm2map(hp.map2alm(mnfgres, lmax=lmax), nside=nside)

# mnspilc = np.load('../../data/band5cmbpsfg/simpilc/pilc_noise_res_map.npy')
# mnshilc = np.load('../../data/band5cmbpsfg/simhilc/hilc_noise_res_map.npy')
# mnsnilc = np.load('../../data/band5cmbpsfg/simnilc/nilc_noise_res_map0.npy')


# cmb = np.load('../../FGSim/CMB/270.npy')
# cmb_b = hp.alm2map(hp.map2alm(cmb, lmax=lmax)[2], nside=512)

# mask = np.load('../mask/north/APOMASKC1_10.npy')

vmin=-0.2
vmax=0.2
hp.orthview(hp.ma(mpilc * mask, badval=0), rot=[100,50,0], half_sky=True, cmap='RdBu', sub=(1,3,1), title='pilc', min=vmin, max=vmax, badcolor='white')
hp.orthview(hp.ma(mhilc * mask, badval=0), rot=[100,50,0], half_sky=True, cmap='RdBu', sub=(1,3,2), title='hilc', min=vmin, max=vmax, badcolor='white')
hp.orthview(hp.ma(mnilc * mask, badval=0), rot=[100,50,0], half_sky=True, cmap='RdBu', sub=(1,3,3), title='nilc', min=vmin, max=vmax, badcolor='white')
plt.show()

vmin=-0.2
vmax=0.2

hp.orthview(hp.ma(mpfgres * mask, badval=0), rot=[100,50,0], half_sky=True, cmap='RdBu', sub=(1,3,1), title='pilc fgres', min=vmin, max=vmax, badcolor='white')
hp.orthview(hp.ma(mhfgres * mask, badval=0), rot=[100,50,0], half_sky=True, cmap='RdBu', sub=(1,3,2), title='hilc fgres', min=vmin, max=vmax, badcolor='white')
hp.orthview(hp.ma(mnfgres * mask, badval=0), rot=[100,50,0], half_sky=True, cmap='RdBu', sub=(1,3,3), title='nilc fgres', min=vmin, max=vmax, badcolor='white')
plt.show()



# clpilc = hp.anafast(mpilc, lmax=lmax)
# clhilc = hp.anafast(mhilc, lmax=lmax)
# clnilc = hp.anafast(mnilc, lmax=lmax)
# clnspilc = hp.anafast(mnspilc, lmax=lmax)
# clnshilc = hp.anafast(mnshilc, lmax=lmax)
# clnsnilc = hp.anafast(mnsnilc, lmax=lmax)
# clcmb = hp.anafast(cmb_b * mask, lmax=lmax)
# plt.plot(l*(l+1)*clcmb/(2*np.pi), label='cmb')
# plt.plot(l*(l+1)*(clpilc-clnspilc)/(2*np.pi), label='pilc')
# plt.plot(l*(l+1)*(clhilc-clnshilc)/(2*np.pi), label='hilc')
# plt.plot(l*(l+1)*(clnilc-clnsnilc)/(2*np.pi), label='nilc')

# plt.legend()
# plt.semilogy()
# plt.show()








