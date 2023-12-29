import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

beam = 63
nside = 2048

max_pixrad = hp.max_pixrad(nside=nside, degrees=True)
print(f'{max_pixrad * 60=}')

vec = hp.ang2vec(theta=0, phi=0, lonlat=True)
ipix_disc = hp.query_disc(nside=nside, vec=vec, radius=0.1 * np.deg2rad(beam)/60)
print(f'{ipix_disc.shape=}')

