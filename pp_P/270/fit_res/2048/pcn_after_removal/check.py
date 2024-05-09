import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../../../../mask/mask_csv/270.csv')

flux_idx = 1
lon = np.rad2deg(df.at[flux_idx, 'lon'])
lat = np.rad2deg(df.at[flux_idx, 'lat'])

mask_list = np.load('./2sigma/mask_1.npy')
print(f'{mask_list=}')

m = np.load('./2sigma/map_q_1.npy')
m_pcn = np.load('../../../../../fitdata/synthesis_data/2048/PSCMBNOISE/270/1.npy')[1]
mask = np.load('../../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
# mask = np.load('../../../../../src/mask/north/BINMASKG2048.npy')

# hp.orthview(m*mask, rot=[100,50,0], half_sky=True, title='after removal')
# hp.orthview(m_pcn*mask, rot=[100,50,0], half_sky=True, title='pcn')

hp.gnomview(m*mask, rot=[lon, lat, 0], title='after fitting')
hp.gnomview(m_pcn*mask, rot=[lon, lat, 0], title='pcn')

plt.show()

