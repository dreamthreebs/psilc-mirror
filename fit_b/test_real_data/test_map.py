import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

ps = np.load('../../fitdata/2048/PS/215/ps.npy')
# ps_b = hp.alm2map(hp.map2alm(ps)[2], nside=2048)

df = pd.read_csv('../../pp_P/mask/mask_csv/215.csv')

flux_idx = 1

lon = np.rad2deg(df.at[flux_idx, 'lon'])
lat = np.rad2deg(df.at[flux_idx, 'lat'])

hp.gnomview(ps[1], rot=[lon, lat, 0])
plt.show()

