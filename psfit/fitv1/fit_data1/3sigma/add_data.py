import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../../../partial_sky_ps/ps_with_nearby/40.csv')

num_ps = len(df)
norm_beam = np.zeros(num_ps)
fit_lon = np.zeros(num_ps)
fit_lat = np.zeros(num_ps)

norm_beam0 = np.load('./norm_beam.npy')
print(f'{norm_beam.shape=}')
norm_beam1 = np.load('./norm_beam1.npy')
print(f'{norm_beam1.shape=}')
norm_beam2 = np.load('./norm_beam2.npy')
print(f'{norm_beam2.shape=}')
norm_beam3 = np.load('./norm_beam3.npy')
print(f'{norm_beam3.shape=}')

norm_beam[0:20] = norm_beam0
norm_beam[20:60] = norm_beam1
norm_beam[60:100] = norm_beam2
norm_beam[100:137] = norm_beam3

fit_lon0 = np.load('./fit_lon.npy')
fit_lon1 = np.load('./fit_lon1.npy')
fit_lon2 = np.load('./fit_lon2.npy')
fit_lon3 = np.load('./fit_lon3.npy')

fit_lon[0:20] = fit_lon0
fit_lon[20:60] = fit_lon1
fit_lon[60:100] = fit_lon2
fit_lon[100:137] = fit_lon3

fit_lat0 = np.load('./fit_lat.npy')
fit_lat1 = np.load('./fit_lat1.npy')
fit_lat2 = np.load('./fit_lat2.npy')
fit_lat3 = np.load('./fit_lat3.npy')

fit_lat[0:20] = fit_lat0
fit_lat[20:60] = fit_lat1
fit_lat[60:100] = fit_lat2
fit_lat[100:137] = fit_lat3

df["fit_norm"] = norm_beam[:]
df["fit_lon"] = fit_lon[:]
df["fit_lat"] = fit_lat[:]
df.to_csv('./40.csv')




