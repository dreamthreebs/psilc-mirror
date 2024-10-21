import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

# m = hp.read_map('./mask.fits')
m_input = hp.read_map('./mask.fits')
# m_output = hp.read_map('../output_inp_b/1.fits')

df = pd.read_csv('../../mask/30.csv')
flux_idx = 1

lon = np.rad2deg(df.at[flux_idx, 'lon'])
lat = np.rad2deg(df.at[flux_idx, 'lat'])

# hp.gnomview(m, rot=[lon, lat, 0], title='mask', xsize=200)
hp.gnomview(m_input, rot=[lon, lat, 0], title='input', xsize=200)
# hp.gnomview(m_output, rot=[lon, lat, 0], title='output', xsize=100)

# hp.orthview(m, half_sky=True)
hp.orthview(m_input, rot=[100,50,0], half_sky=True)

plt.show()



