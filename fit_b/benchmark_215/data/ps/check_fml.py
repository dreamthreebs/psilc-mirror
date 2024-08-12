import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../../mask/215.csv')
m = np.load('./ps.npy')


flux_idx = 0
lon = np.rad2deg(df.at[flux_idx, 'lon'])
lat = np.rad2deg(df.at[flux_idx, 'lat'])

hp.gnomview(m[1], rot=[lon, lat, 0])
hp.gnomview(m[2], rot=[lon, lat, 0])
plt.show()
