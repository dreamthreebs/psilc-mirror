import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../../../../partial_sky_ps/ps_with_nearby/40.csv')
flux_idx = 4
ps_lon = np.rad2deg(df.at[flux_idx, 'lon'])
ps_lat = np.rad2deg(df.at[flux_idx, 'lat'])

mask = np.load('../../../../../src/mask/north/BINMASKG2048.npy')
m = hp.read_map('./after_inpainting_map.fits', field=0)
# hp.mollview(m)
hp.orthview(m * mask, rot=[100,50,0], half_sky=True)
hp.gnomview(m * mask, rot=[ps_lon, ps_lat, 0], title='inpainted result')
plt.show()
