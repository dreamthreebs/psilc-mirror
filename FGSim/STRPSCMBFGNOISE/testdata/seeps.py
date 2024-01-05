import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../../../psfit/partial_sky_ps/ps_with_nearby/40.csv')
m = np.load('./40.npy')[0]

flux_idx = 1
lon = np.rad2deg(df.at[flux_idx, 'lon'])
lat = np.rad2deg(df.at[flux_idx, 'lat'])

hp.gnomview(m, rot=[lon, lat, 0], title='cmb + fg + noise')
plt.show()



