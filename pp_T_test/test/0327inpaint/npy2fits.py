import numpy as np
import healpy as hp

# ps_cmb_noise = np.load('./ps_cmb_noise.npy')
# cmb_noise = np.load('./cmb_noise.npy')
# mask = np.load('./mask2.npy')
cmb = np.load('../../../fitdata/2048/CMB/155/0.npy')[0]

# hp.write_map('./fits_file/ps_cmb_noise.fits', ps_cmb_noise, overwrite=True)
# hp.write_map('./fits_file/mask2.fits', mask, overwrite=True)
# hp.write_map('./fits_file/cmb_noise.fits', cmb_noise, overwrite=True)
hp.write_map('./fits_file/cmb.fits', cmb, overwrite=True)
