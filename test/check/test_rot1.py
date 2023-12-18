import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../ps_sort/sort_by_iflux/40.csv')

# lon = np.rad2deg(df.at[44,'lon'])
# lat = np.rad2deg(df.at[44,'lat'])

lon = 30
lat = 60

lon_test = 31
lat_test = 61


vec_rot = hp.ang2vec(theta=lon, phi=lat, lonlat=True)
print(f'{vec_rot=}')

vec_test = hp.ang2vec(theta=lon_test, phi=lat_test, lonlat=True)
print(f'{vec_test=}')

r = hp.rotator.Rotator(rot=[lon, lat, 0], deg=True)
vec_rot4 = r(vec_test)
print(f'{vec_rot4=}')


r = hp.rotator.Rotator(rot=[lon, lat, 0], deg=True)
vec_rot5 = r.I(vec_rot4)

print(f'{vec_rot5=}')





