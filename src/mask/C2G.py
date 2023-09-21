''' Get binary mask at Galaxy coordinate from NSTD map '''
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

m = np.load('../../FGSim/northMask.npy') # Celestial map
mask = m > 0
bin_mask = np.zeros_like(m)
bin_mask[mask] = 1

r = hp.Rotator(coord=['C', 'G'])
bin_mask_G = r.rotate_map_pixel(bin_mask)
bin_maskG = np.where(bin_mask_G>0,1.,0.)

hp.mollview(bin_maskG);plt.show()
np.save('./north/BINMASKG.npy', bin_maskG)


