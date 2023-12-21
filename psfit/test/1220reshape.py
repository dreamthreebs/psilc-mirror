import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

nside = 2048

ipix_disc = hp.query_disc(nside=nside, vec=(0,0,0), radius=np.deg2rad(1))
print(f'{ipix_disc=}')
print(f'{ipix_disc.shape=}')

vec = hp.pix2vec(nside=nside, ipix=ipix_disc)
print(f'{vec=}')
vec_arr = np.array(vec)
print(f'{vec_arr.shape=}')

vec_reshape = np.reshape(vec, (3,-1))
print(f'{vec_arr.shape=}')


