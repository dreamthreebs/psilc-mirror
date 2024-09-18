import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

nside = 2048
beam = 67

df = pd.read_csv('./30_bak.csv')
print(f'{len(df)=}')

lon_lat_list = []
flux_idx_list = []

first_lon = np.rad2deg(df.at[0, 'lon'])
first_lat = np.rad2deg(df.at[0, 'lat'])

lon_lat_list.append((first_lon, first_lat))
flux_idx_list.append(0)

for flux_idx in range(1, len(df)):
    rank = df.at[flux_idx, 'rank']
    print(f'{rank=}')
    lon = np.rad2deg(df.at[flux_idx, 'lon'])
    lat = np.rad2deg(df.at[flux_idx, 'lat'])
    dir_1 = (lon, lat)
    dir_2 = np.asarray(lon_lat_list).T

    angle = np.rad2deg(hp.rotator.angdist(dir1=dir_1, dir2=dir_2, lonlat=True))
    min_angle = np.min(angle)

    if min_angle < 3.05 * beam / 60:
        continue

    lon_lat_list.append((lon, lat))
    flux_idx_list.append(flux_idx)


print(f'{len(lon_lat_list)=}')
print(f'{flux_idx_list=}')

df_filtered = df.iloc[flux_idx_list]
df_filtered = df_filtered.drop(columns=['rank'])
df_filtered = df_filtered.reset_index(drop=True)

df_filtered.to_csv(f'filtered_data.csv', index=True)
