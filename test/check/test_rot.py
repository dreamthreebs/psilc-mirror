import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../ps_sort/sort_by_iflux/40.csv')
lon = np.rad2deg(df.at[44,'lon'])
lat = np.rad2deg(df.at[44,'lat'])

# lon = 55
# lat = 60



vec_rot = hp.ang2vec(theta=lon, phi=lat, lonlat=True)
print(f'{vec_rot=}')

vec_init = (0,0,1)

r = hp.rotator.Rotator(rot=[55, -30, 0], deg=True, inv=True)
vec_rot2 = r(vec_init)

print(f'{vec_rot2=}')
r = hp.rotator.Rotator(rot=[0, 60, 55], deg=True)
vec_rot3 = r(vec_init)
print(f'{vec_rot3=}')

r = hp.rotator.Rotator(rot=[lon, lat, 0], deg=True)
vec_rot4 = r(vec_rot)
print(f'{vec_rot4=}')


r = hp.rotator.Rotator(rot=[lon, lat, 0], deg=True)
vec_rot5 = r.I(vec_rot4)

print(f'{vec_rot5=}')





rotmat = hp.rotator.euler_matrix_new(lon, lat,0 , deg=True, ZYX=True)
vec_rot1 = hp.rotator.rotateVector(rotmat=rotmat, vec=vec_init)
print(f'{vec_rot1=}')
vec_rot1_1 = hp.rotator.rotateVector(rotmat=np.linalg.inv(rotmat), vec=vec_rot1)
print(f'{vec_rot1_1=}')

vec_init = (1,0,0)
r = hp.rotator.Rotator(rot=[90, 0, 90], deg=True)
vec_rot3 = r(vec_init)

print(f'{vec_rot3=}')


