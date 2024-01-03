import numpy as np
import pandas as pd
import healpy as hp
import matplotlib.pyplot as plt

norm_beam = np.load('./norm_beam1.npy')
lon = np.load('./fit_lon1.npy')
lat = np.load('./fit_lat1.npy')
print(f'{norm_beam.shape=}')
print(f'{norm_beam=}')

def flux2norm_beam(flux):
    # from mJy to muK_CMB to norm_beam
    coeffmJy2norm = 2.1198465131100624e-05
    return coeffmJy2norm * flux

df = pd.read_csv('../partial_sky_ps/ps_with_nearby/40.csv')
norm_beam_true = flux2norm_beam(df["iflux"])
norm_beam_true = norm_beam_true.to_numpy()

norm_beam_all = np.zeros_like(norm_beam_true)
norm_beam_all[:norm_beam.shape[0]] = norm_beam
lon_all = np.zeros_like(norm_beam_true)
lon_all[:lon.shape[0]] = lon
lat_all = np.zeros_like(norm_beam_true)
lat_all[:lat.shape[0]] = lat

fit_error = np.abs(norm_beam_all - norm_beam_true) / norm_beam_true
fit_error = np.where(fit_error != 1.0, fit_error, 0)

df["fit_norm"] = norm_beam_all
df["fit_error"] = fit_error
df["fit_lon"] = lon_all
df["fit_lat"] = lat_all

df.to_csv(f'./data/40.csv', index=False)



