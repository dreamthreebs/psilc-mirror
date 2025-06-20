import numpy as np
import healpy as hp

m = hp.read_map(f'./HFI_SkyMap_353-psb_2048_R3.01_full.fits', field=(0,1,2))

print(f'{m.shape=}')

m_out = hp.ud_grade(m, nside_out=512)
np.save('./north_353.npy', m_out)
