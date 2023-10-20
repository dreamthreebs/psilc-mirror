import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('./pilc_fgres_map1.npy')
hp.orthview(hp.ma(m, badval=0), rot=[100,50,0], half_sky=True, cmap='RdBu', badcolor='white', min=-0.8, max=0.8)
plt.show()


