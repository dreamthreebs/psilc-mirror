import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

def calc_avg_var():

    m_list = []
    
    for rlz_idx in range(100):
        print(f'{rlz_idx=}')
        m_rlz = np.load(f'../ps_cmb_noise_residual/2sigma/map{rlz_idx}.npy')
        m_list.append(m_rlz)
    
    m_avg = np.sum(np.asarray(m_list), axis=0) / 100
    m_var = np.var(np.asarray(m_list), axis=0)

    # np.save('./pcn/2sigma/ps_res_avg.npy', m_avg)
    np.save('./pcn/2sigma/ps_res_var.npy', m_var)

# calc_avg_var()

m_avg = np.load('./pcn/2sigma/ps_res_avg.npy')
m_var = np.load('./pcn/2sigma/ps_res_var.npy')
m_one_rlz = np.load(f'../ps_cmb_noise_residual/2sigma/map3.npy')

flux_idx = 2

df = pd.read_csv('../../../../partial_sky_ps/ps_in_mask/2048/40mask.csv')
lon = np.rad2deg(df.at[flux_idx,'lon'])
lat = np.rad2deg(df.at[flux_idx,'lat'])

hp.gnomview(m_avg, rot=[lon, lat, 0], title=f'ps_res avg, {flux_idx=}')
hp.gnomview(m_var, rot=[lon, lat, 0], title=f'ps_res var, {flux_idx=}')
hp.gnomview(m_one_rlz, rot=[lon, lat, 0], title=f'ps_res one rlz, {flux_idx=}')
plt.show()




