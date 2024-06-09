import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import pymaster as nmt

nside = 512
df = pd.read_csv(f'../../../../../mask/mask_csv/270.csv')
beam = 9

# orimask_2048 = np.load('../../../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
# orimask = hp.ud_grade(orimask_2048, nside_out=512 )
# orimask[orimask<1] = 0
# hp.orthview(orimask, rot=[100,50,0], half_sky=True)
# plt.show()

# np.save('./mask/bin_mask.npy', orimask)

orimask = np.load('./mask/bin_mask.npy')
apo_edge_mask = nmt.mask_apodization(orimask, aposize=3)
hp.orthview(apo_edge_mask, rot=[100,50,0], half_sky=True)
plt.show()
np.save('./mask/apo_edge_mask.npy', apo_edge_mask)

mask_list = np.load(f'../../pcn_after_removal/3sigma/mask_1.npy')
mask = np.ones_like(orimask)


for flux_idx in mask_list:
    print(f'{flux_idx=}')
    lon = np.rad2deg(df.at[flux_idx, 'lon'])
    lat = np.rad2deg(df.at[flux_idx, 'lat'])

    ctr_vec = hp.ang2vec(theta=lon, phi=lat, lonlat=True)
    ipix_mask = hp.query_disc(nside=nside, vec=ctr_vec, radius=1.5 * np.deg2rad(beam) / 60)
    mask[ipix_mask] = 0

hp.orthview(mask, rot=[100,50, 0], title='mask', xsize=2000)
plt.show()

hp.orthview(mask*orimask, rot=[100, 50, 0], title='bin_final_mask', xsize=2000)
plt.show()
np.save('./mask/final_bin_mask.npy', mask*orimask)

apo_ps_mask = nmt.mask_apodization(mask, aposize=1) * apo_edge_mask
hp.orthview(apo_ps_mask, rot=[100,50, 0], title='mask', xsize=2000)
plt.show()

np.save(f'./mask/final_mask.npy', apo_ps_mask)

