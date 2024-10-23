import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

nside = 2048

flux_idx = 2
df = pd.read_csv('./30.csv')
lon = np.rad2deg(df.at[flux_idx, 'lon'])
lat = np.rad2deg(df.at[flux_idx, 'lat'])

pix_idx = hp.ang2pix(nside, theta=lon, phi=lat, lonlat=True)
pix_lon, pix_lat = hp.pix2ang(nside, ipix=pix_idx, lonlat=True)

print(f'{lon=}, {pix_lon=}')
print(f'{lat=}, {pix_lat=}')

