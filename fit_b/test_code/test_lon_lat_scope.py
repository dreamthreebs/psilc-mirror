import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nside = 1024
theta = np.linspace(0,np.pi,100)
phi = np.linspace(0,2*np.pi,100)

ipix = hp.ang2pix(nside=nside, theta=theta, phi=phi)
lon, lat = hp.pix2ang(ipix=ipix, nside=nside, lonlat=True)
print(f'{lon=}, {lat=}')
