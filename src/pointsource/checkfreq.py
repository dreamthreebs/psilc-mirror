import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.io import readsav
from astropy import units as u




imap_out = hp.read_map('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/30GHz/strongirps_map_30GHz.fits', field=(0,1,2))[0]


imap_out_270 = hp.read_map('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/270GHz/strongirps_map_270GHz.fits', field=(0,1,2))[0]


imap_out[imap_out!=0] = 1
imap_out_270[imap_out_270!=0] = 1

hp.mollview(imap_out, title='out')
hp.mollview(imap_out-imap_out_270, title='difference between 30 and 270')

# difference_nonzero = np.count_nonzero(imap-imap_out)
difference_nonzero = np.count_nonzero(imap_out-imap_out_270)
plt.show()
print(f'{difference_nonzero=}')


