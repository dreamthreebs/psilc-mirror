import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

mpilc = np.load('../../data/band5/simpilc/pilc_map.npy')
mhilc = np.load('../../data/band5/simhilc/hilc_map.npy')
mnilc = np.load('../../data/band5/simnilc/nilc_map0.npy')

mpfgres = np.load('../../data/band5/simpilc/pilc_fgres_map.npy')
mhfgres = np.load('../../data/band5/simhilc/hilc_fgres_map.npy')
mnfgres = np.load('../../data/band5/simnilc/nilc_fgres_map0.npy')

mask = np.load('../mask/north/BINMASKG.npy')

vmin=-0.8
vmax=0.8
hp.orthview(hp.ma(mpilc * mask, badval=0), badcolor='white', rot=[100,50,0], half_sky=True, cmap='RdBu', sub=(1,3,1), title='pilc', min=vmin, max=vmax)
hp.orthview(hp.ma(mhilc * mask, badval=0), badcolor='white', rot=[100,50,0], half_sky=True, cmap='RdBu', sub=(1,3,2), title='hilc', min=vmin, max=vmax)
hp.orthview(hp.ma(mnilc * mask, badval=0), badcolor='white', rot=[100,50,0], half_sky=True, cmap='RdBu', sub=(1,3,3), title='nilc', min=vmin, max=vmax)
plt.show()
vmin=-0.2
vmax=0.2
hp.orthview(hp.ma(mpfgres * mask, badval=0), badcolor='white', rot=[100,50,0], half_sky=True, cmap='RdBu', sub=(1,3,1), title='pilc fgres', min=vmin, max=vmax)
hp.orthview(hp.ma(mhfgres * mask, badval=0),  badcolor='white',rot=[100,50,0], half_sky=True, cmap='RdBu', sub=(1,3,2), title='hilc fgres', min=vmin, max=vmax)
hp.orthview(hp.ma(mnfgres * mask, badval=0), badcolor='white', rot=[100,50,0], half_sky=True, cmap='RdBu', sub=(1,3,3), title='nilc fgres', min=vmin, max=vmax)
plt.show()


