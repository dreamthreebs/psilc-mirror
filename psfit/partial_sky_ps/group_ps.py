import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

nside = 2048
mask = np.load('../../src/mask/north/BINMASKG2048.npy')
df = pd.read_csv('./ps_in_mask/mask40.csv')

all_angles_list = []
for i in range(len(df)):
    angles_list = []
    lon_0 = np.rad2deg(df.at[i,'lon'])
    lat_0 = np.rad2deg(df.at[i,'lat'])
    dir_0 = (lon_0, lat_0)

    bool_arr = df.index != i
    # print(f'{bool_arr=}')
    lon = np.rad2deg(df.loc[bool_arr,'lon'])
    lat = np.rad2deg(df.loc[bool_arr,'lat'])
    dir_other = (lon, lat)
    ang = hp.rotator.angdist(dir1=dir_0, dir2=dir_other, lonlat=True)
    ang_deg = np.rad2deg(np.min(ang))
    print(f'{ang_deg=}')
# print(f'{angles}')
# print(f'{np.min(angles)}')
