import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

nside = 2048
mask = np.load('../../src/mask/north/BINMASKG2048.npy')
df = pd.read_csv('../../test/ps_sort/sort_by_iflux/40.csv', index_col=0)

def in_or_out_mask(row):
    lon = row['lon']
    lat = row['lat']
    ipix = hp.ang2pix(nside=nside, theta=np.rad2deg(lon), phi=np.rad2deg(lat), lonlat=True)
    if mask[ipix] == 1.0:
        return True
    elif mask[ipix] == 0.0:
        return False
    else:
        raise ValueError('mask data type should be float!')

condition = df.apply(in_or_out_mask, axis=1)

filtered_data = df[condition]

filtered_data.to_csv('./ps_in_mask/mask40.csv', index=True)

