import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

m_sum = np.zeros(hp.nside2npix(2048))

for rlz_idx in range(100):
    print(f'{rlz_idx=}')
    m_rlz = np.load(f'../ps_cmb_noise_residual/2sigma/map{rlz_idx}.npy')
    m_sum = m_sum + m_rlz

m_avg = m_sum / 100

df = pd.read_csv('../../../../partial_sky_ps/ps_in_mask/2048/40mask.csv')
flux_idx = 1
lon = np.rad2deg(df.at[flux_idx,'lon'])
lat = np.rad2deg(df.at[flux_idx,'lat'])

np.save('./pcn/2sigma/ps_res_avg.npy', m_avg)

hp.gnomview(m_avg, rot=[lon, lat, 0], title=f'ps_res avg, {flux_idx=}')
plt.show()




