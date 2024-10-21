import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

flux_idx = 0
df = pd.read_csv('../mask/30.csv')

lon = np.rad2deg(df.at[flux_idx, 'lon'])
lat = np.rad2deg(df.at[flux_idx, 'lat'])

m = np.load('./pcfn/0.npy')
hp.gnomview(m[1], rot=[lon, lat, 0], min=-20,max=20, title='pcfn')
m = np.load('./ps/ps.npy')
hp.gnomview(m[1], rot=[lon, lat, 0], min=-20,max=20, title='ps')
m = np.load('./fg/0.npy')
hp.gnomview(m[1], rot=[lon, lat, 0], min=-20,max=20, title='fg')
m = np.load('./cmb/0.npy')
hp.gnomview(m[1], rot=[lon, lat, 0], min=-20,max=20, title='cmb')



plt.show()



