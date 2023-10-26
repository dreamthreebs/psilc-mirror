import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

mask = np.load('../mask/north/APOMASKC1_10.npy')
mpilc = np.load('../../data/noapo/simpilc/pilc_map.npy') * mask
mhilc = np.load('../../data/noapo/simhilc/hilc_map.npy') * mask
mnilc = np.load('../../data/noapo/simnilc/nilc_map0.npy') * mask

mpfgres = np.load('../../data/noapo/simpilc/pilc_fgres_map.npy') * mask
mhfgres = np.load('../../data/noapo/simhilc/hilc_fgres_map.npy') * mask
mnfgres = np.load('../../data/noapo/simnilc/nilc_fgres_map0.npy') * mask

vmin=-0.8
vmax=0.8
hp.orthview(mpilc, rot=[100,50,0], half_sky=True, cmap='RdBu', sub=(1,3,1), title='pilc', min=vmin, max=vmax)
hp.orthview(mhilc, rot=[100,50,0], half_sky=True, cmap='RdBu', sub=(1,3,2), title='hilc', min=vmin, max=vmax)
hp.orthview(mnilc, rot=[100,50,0], half_sky=True, cmap='RdBu', sub=(1,3,3), title='nilc', min=vmin, max=vmax)
plt.show()

vmin=-0.2
vmax=0.2

hp.orthview(mpfgres, rot=[100,50,0], half_sky=True, cmap='RdBu', sub=(1,3,1), title='pilc fgres', min=vmin, max=vmax)
hp.orthview(mhfgres, rot=[100,50,0], half_sky=True, cmap='RdBu', sub=(1,3,2), title='hilc fgres', min=vmin, max=vmax)
hp.orthview(mnfgres, rot=[100,50,0], half_sky=True, cmap='RdBu', sub=(1,3,3), title='nilc fgres', min=vmin, max=vmax)
plt.show()


