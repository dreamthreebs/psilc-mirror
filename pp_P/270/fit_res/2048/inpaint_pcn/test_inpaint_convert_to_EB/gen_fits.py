import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

freq = 155
cn = np.load(f'../../../../../../fitdata/synthesis_data/2048/CMBNOISE/{freq}/1.npy')
ori_mask = np.load('../../../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')

hp.write_map('./cn_Q.fits', cn[1], overwrite=True)
hp.write_map('./cn_U.fits', cn[2], overwrite=True)
hp.write_map('./mask.fits', ori_mask, overwrite=True)
