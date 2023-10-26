import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

mpilc = np.load('../../data/sim300/sim300pilc/pilc_map.npy')
mhilc = np.load('../../data/test/cmbfghilc/hilc_map.npy')
mnilc = np.load('../../data/test/cmbfgnilc/nilc_map0.npy')

mpfgres = np.load('../../data/test/cmbfgpilc/pilc_fgres_map.npy')
mhfgres = np.load('../../data/test/cmbfghilc/hilc_fgres_map.npy')
mnfgres = np.load('../../data/test/cmbfgnilc/nilc_fgres_map0.npy')


vmin=-0.8
vmax=0.8
hp.orthview(mpilc, rot=[100,50,0], half_sky=True, cmap='RdBu', sub=(1,3,1), title='pilc', min=vmin, max=vmax)
hp.orthview(mhilc, rot=[100,50,0], half_sky=True, cmap='RdBu', sub=(1,3,2), title='hilc', min=vmin, max=vmax)
hp.orthview(mnilc, rot=[100,50,0], half_sky=True, cmap='RdBu', sub=(1,3,3), title='nilc', min=vmin, max=vmax)
plt.show()
vmin=-0.1
vmax=0.1
hp.orthview(mpfgres, rot=[100,50,0], half_sky=True, cmap='RdBu', sub=(1,3,1), title='pilc fgres', min=vmin, max=vmax)
hp.orthview(mhfgres, rot=[100,50,0], half_sky=True, cmap='RdBu', sub=(1,3,2), title='hilc fgres', min=vmin, max=vmax)
hp.orthview(mnfgres, rot=[100,50,0], half_sky=True, cmap='RdBu', sub=(1,3,3), title='nilc fgres', min=vmin, max=vmax)
plt.show()


