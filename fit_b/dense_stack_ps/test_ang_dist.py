import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

dir1 = (112.66,65.74)
dir2 = (119.94,67.56)

distance = np.rad2deg(hp.rotator.angdist(dir1=dir1, dir2=dir2, lonlat=True))
print(f'{distance=}')

nside = 2048
print(hp.ang2pix(theta=45, phi=30, lonlat=True, nside=nside))
print(hp.ang2pix(theta=45+360, phi=30, lonlat=True, nside=nside))
print(hp.ang2pix(theta=45+720, phi=-100, lonlat=True, nside=nside))


