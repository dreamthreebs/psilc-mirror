import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
import ipdb

freq = 270
flux_idx = 1
lmax = 1999
nside = 2048

df = pd.read_csv(f'../../../mask/mask_csv/{freq}.csv')
lon = np.rad2deg(df.at[flux_idx, 'lon'])
lat = np.rad2deg(df.at[flux_idx, 'lat'])

pcn = np.load(f'../../../../fitdata/synthesis_data/2048/PSCMBNOISE/{freq}/1.npy')
cn = np.load(f'../../../../fitdata/synthesis_data/2048/CMBNOISE/{freq}/1.npy')

pcn_B = hp.alm2map(hp.map2alm(pcn, lmax=lmax)[2], nside=nside)
pcn_E = hp.alm2map(hp.map2alm(pcn, lmax=lmax)[1], nside=nside)

cn_B = hp.alm2map(hp.map2alm(cn, lmax=lmax)[2], nside=nside)
cn_E = hp.alm2map(hp.map2alm(cn, lmax=lmax)[1], nside=nside)

rmv_E = np.load(f'./pcn_after_removal/2sigma/E/map_crp_e1.npy')
rmv_B = np.load(f'./pcn_after_removal/2sigma/B/map_cln_b1.npy')

fig_size = 60

m_proj_cn = hp.gnomview(cn_B, rot=[lon, lat, 0], title='cn B', xsize=fig_size, return_projected_map=True)
m_proj_cn = hp.gnomview(pcn_B-cn_B, rot=[lon, lat, 0], title='pcn - cn B', xsize=fig_size, return_projected_map=True)
vmin = 1.5 * np.min(m_proj_cn)
vmax = 1.5 * np.max(m_proj_cn)
plt.show()

hp.gnomview(pcn_B, rot=[lon, lat, 0], title='pcn B', xsize=fig_size, sub=221, min=vmin, max=vmax)
hp.gnomview(cn_B, rot=[lon, lat, 0], title='cn B', xsize=fig_size, sub=222, min=vmin, max=vmax)
hp.gnomview(rmv_B, rot=[lon, lat, 0], title='removal B', xsize=fig_size, sub=223, min=vmin, max=vmax)
hp.gnomview(rmv_B - cn_B, rot=[lon, lat, 0], title='residual B', xsize=fig_size, sub=224)
plt.show()


m_proj_cn = hp.gnomview(cn_E, rot=[lon, lat, 0], title='cn E', xsize=fig_size, return_projected_map=True)
vmin = 1.5 * np.min(m_proj_cn)
vmax = 1.5 * np.max(m_proj_cn)
plt.show()

hp.gnomview(pcn_E, rot=[lon, lat, 0], title='pcn E', xsize=fig_size, sub=221, min=vmin, max=vmax)
hp.gnomview(cn_E, rot=[lon, lat, 0], title='cn E', xsize=fig_size, sub=222, min=vmin, max=vmax)
hp.gnomview(rmv_E, rot=[lon, lat, 0], title='removal E', xsize=fig_size, sub=223, min=vmin, max=vmax)
hp.gnomview(rmv_E- cn_E, rot=[lon, lat, 0], title='residual E', xsize=fig_size, sub=224)
plt.show()

