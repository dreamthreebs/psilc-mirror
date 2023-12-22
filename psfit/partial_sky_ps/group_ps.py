import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

nside = 2048
beam = 63
mask = np.load('../../src/mask/north/BINMASKG2048.npy')
df = pd.read_csv('./ps_in_mask/mask40.csv')
df1 = pd.read_csv('../../test/ps_sort/sort_by_iflux/40.csv')

all_angles_list = []
sum_nearby = 0
n_nearby_arr = np.zeros(len(df))
for i in range(len(df)):
    angles_list = []
    lon_0 = np.rad2deg(df.at[i,'lon'])
    lat_0 = np.rad2deg(df.at[i,'lat'])
    dir_0 = (lon_0, lat_0)

    bool_arr = df1.loc[:,'Unnamed: 0'] != df.at[i,'Unnamed: 0']
    # print(f'{bool_arr=}')
    lon = np.rad2deg(df1.loc[bool_arr,'lon'])
    lat = np.rad2deg(df1.loc[bool_arr,'lat'])
    dir_other = (lon, lat)
    ang = np.rad2deg(hp.rotator.angdist(dir1=dir_0, dir2=dir_other, lonlat=True))
    threshold_1 = 2.2 * beam / 60
    threshold_2 = 1.1 * beam / 60
    if np.min(ang) > threshold_1:
        print(f'this point source is single')
    elif np.min(ang) > threshold_2: 
        # ang = ang[np.nonzero(np.where((ang > threshold_2) & (ang < threshold_1), ang, 0))]
        ang = ang[np.nonzero(np.where((ang < threshold_1), ang, 0))]
        # print(f'have nearby point sources > 1.1 beam size, {ang=}')
        # print(f'have nearby point sources > 1.1 beam size, {ang.shape=}')
        print(f'have nearby point sources > 1.1 beam size, {len(ang)=}')
        n_nearby_arr[i] = len(ang)
        sum_nearby += len(ang)
    else:
        ang = ang[np.nonzero(np.where((ang < threshold_1), ang, 0))]
        # print(f'have nearby point sources < 1.1 beamsize, {ang=}')
        print(f'have nearby point sources < 1.1 beamsize, {ang.shape=}')
        n_nearby_arr[i] = len(ang)
        sum_nearby += len(ang)

    # ang_deg = np.rad2deg(np.min(ang))
    # print(f'{ang_deg=}')

print(f'{sum_nearby/len(df)=}')
df['number_nearby'] = n_nearby_arr
df.to_csv(f'./40.csv', index=False)

# print(f'{angles}')
# print(f'{np.min(angles)}')
