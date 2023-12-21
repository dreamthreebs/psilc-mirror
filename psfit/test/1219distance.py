import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

beam = 63 # arcmin
sigma = np.deg2rad(beam)/60 / (np.sqrt(8*np.log(2)))
print(f'{sigma=}')

nside = 2048

# m = np.load('../../FGSim/STRPSCMBFGNOISE/40.npy')[0]
# m = np.load('../../FGSim/STRPSFGNOISE/40.npy')[0]
# m = np.load('../../FGSim/STRPSCMBNOISE/40.npy')[0]
m = np.load('../../FGSim/PSNOISE/2048/40.npy')[0]

df = pd.read_csv('../../test/ps_sort/sort_by_iflux/40.csv')
lon = df.at[44, 'lon']
lat = df.at[44, 'lat']
iflux = df.at[44, 'iflux']

# hp.gnomview(m, rot=[np.rad2deg(lon), np.rad2deg(lat), 0])
# plt.show()

itp_val = hp.get_interp_val(m, theta=np.rad2deg(lon), phi=np.rad2deg(lat), lonlat=True)
print(f'{itp_val=}')

max_pix_arcmin = hp.max_pixrad(nside=nside, degrees=True) * 60
print(f'{max_pix_arcmin=}')

dir1 = hp.pix2ang(nside=nside, ipix=231542, lonlat=True)
vec1 = np.array(hp.pix2vec(nside=nside, ipix=231542))
dir2 = hp.pix2ang(nside=nside, ipix=31212782, lonlat=True)
vec2 = np.array(hp.pix2vec(nside=nside, ipix=31212782))
angles1 = np.arccos(vec1 @ vec2)
print(f'{angles1=}')

angles2 = hp.rotator.angdist(dir1=dir1, dir2=dir2, lonlat=True)
print(f'{angles2=}')


