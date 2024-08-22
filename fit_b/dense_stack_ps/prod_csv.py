import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

from test_mJy_to_muKCMB import mJy_to_uKCMB

nside = 2048
beam = 67
freq = 30
nside2pixarea_factor = hp.nside2pixarea(nside=nside)

ps_lon_lat = np.load('./data/ps_lon_lat.npy')
print(f'{ps_lon_lat=}')

lon = np.deg2rad(ps_lon_lat[:,0])
lat = np.deg2rad(ps_lon_lat[:,1])
print(f'{lon=}')
print(f'{lat=}')

n_ps = len(lon)


seed = 4242
rng = np.random.default_rng(seed)

P = 10000
phi = rng.uniform(0, 2*np.pi, size=n_ps)

Delta_Q = P * np.cos(phi)
Delta_U = P * np.sin(phi)
print(f'{Delta_Q=}')
print(f'{Delta_U=}')
print(f'{Delta_Q**2 + Delta_U**2=}')

qflux = Delta_Q / (mJy_to_uKCMB(1, freq) / nside2pixarea_factor)
uflux = Delta_U / (mJy_to_uKCMB(1, freq) / nside2pixarea_factor)
print(f'{qflux=}')
print(f'{uflux=}')

pflux = np.sqrt(qflux**2 + uflux**2)

df = pd.DataFrame({
    "lon": lon.astype('float64'),
    "lat": lat.astype('float64'),
    "qflux": qflux.astype('float64'),
    "uflux": uflux.astype('float64'),
    "pflux": pflux.astype('float64')
        }
        )

df.to_csv(f'./mask/{freq}.csv', index=True)

