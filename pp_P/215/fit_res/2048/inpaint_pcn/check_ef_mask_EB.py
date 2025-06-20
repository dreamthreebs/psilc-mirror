import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

freq = 215
threshold = 3
rlz_idx = 1
lmax = 1999
nside = 2048

flux_idx = 1
df = pd.read_csv(f'../../../../mask/mask_csv/{freq}.csv')
lon = np.rad2deg(df.at[flux_idx, 'lon'])
lat = np.rad2deg(df.at[flux_idx, 'lat'])

cn = np.load(f'../../../../../fitdata/synthesis_data/2048/CMBNOISE/{freq}/{rlz_idx}.npy')
mask = hp.read_map(f'./{threshold}sigma/bin_mask/{rlz_idx}.fits')

alm_cn_i, alm_cn_e, alm_cn_b = hp.map2alm(cn, lmax=lmax)
cn_e = hp.alm2map(alm_cn_e, nside=nside)
cn_b = hp.alm2map(alm_cn_b, nside=nside)

ps_mask_E = hp.read_map(f'./{threshold}sigma/EB/E_input/{rlz_idx}.fits')
ps_mask_B = hp.read_map(f'./{threshold}sigma/EB/B_input/{rlz_idx}.fits')

ps_inp_E = hp.read_map(f'./{threshold}sigma/EB/E_output/{rlz_idx}.fits')
ps_inp_B = hp.read_map(f'./{threshold}sigma/EB/B_output/{rlz_idx}.fits')

hp.gnomview(cn_e, rot=[lon,lat,0], title='cn_e')
hp.gnomview(cn_b, rot=[lon,lat,0], title='cn_b')
hp.gnomview(ps_mask_E, rot=[lon,lat,0], title='ps mask e')
hp.gnomview(ps_mask_B, rot=[lon,lat,0], title='ps mask b')
hp.gnomview(ps_inp_E, rot=[lon,lat,0], title='ps inp e')
hp.gnomview(ps_inp_B, rot=[lon,lat,0], title='ps inp b')

# hp.gnomview((cn_e-ps_mask_E)*mask, rot=[lon,lat,0], title='cn_e - ps_mask_e')
# hp.gnomview((cn_b-ps_mask_B)*mask, rot=[lon,lat,0], title='cn_b - ps_mask_b')

plt.show()




