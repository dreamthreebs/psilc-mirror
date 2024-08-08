import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

m = hp.read_map('./mask_1.fits')
m_input = hp.read_map('../input_b/0.fits')
# m_output = hp.read_map('../output_inp_b/1.fits')

df = pd.read_csv('../../mask/215.csv')
flux_idx = 1

lon = np.rad2deg(df.at[flux_idx, 'lon'])
lat = np.rad2deg(df.at[flux_idx, 'lat'])

hp.gnomview(m, rot=[lon, lat, 0], title='mask', xsize=100)
hp.gnomview(m_input, rot=[lon, lat, 0], title='input', xsize=100)
# hp.gnomview(m_output, rot=[lon, lat, 0], title='output', xsize=100)

# hp.mollview(m_input)
# hp.mollview(m_output)

plt.show()

