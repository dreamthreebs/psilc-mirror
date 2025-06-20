import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

freq = 215
m = hp.read_map('./3sigma/bin_mask/1.fits')
mask = np.load('../pcfn_after_removal/3sigma/mask_1.npy')
print(f"{mask=}")

flux_idx = 1

df = pd.read_csv(f'../../../../mask/mask_csv/{freq}.csv')
lon = np.rad2deg(df.at[flux_idx, 'lon'])
lat = np.rad2deg(df.at[flux_idx, 'lat'])

hp.gnomview(m, rot=[lon,lat,0])
plt.show()



