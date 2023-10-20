from ftbutils.plot_healpy_maps import *

m = np.load('./30.npy')
plot_scalar_map_mollview(m[0])
