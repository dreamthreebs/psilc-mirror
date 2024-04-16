import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd

fig_size = 60
freq = 145
nstd = np.load(f'../../../../FGSim/NSTDNORTH/2048/{freq}.npy')[0][0]
res_max = nstd * 5

def cpr_ps_cmb_noise(df):
    m_res = np.load('./ps_cmb_noise_residual/2sigma/map0.npy')
    mask_list = np.load('./ps_cmb_noise_residual/2sigma/mask0.npy')
    print(f'{mask_list=}')

    m_noise = np.load(f'../../../../fitdata/synthesis_data/2048/CMBNOISE/{freq}/0.npy')[0].copy()
    m_removal = m_res + m_noise

    m_inpaint = hp.read_map('./INPAINT/output/pcn/2sigma/0.fits', field=0)
    m_origin = np.load(f'../../../../fitdata/synthesis_data/2048/PSCMBNOISE/{freq}/0.npy')[0].copy()

    m_inpaint_res = m_inpaint - m_noise
    m_removal_res = m_removal - m_noise

    # hp.orthview(m_removal, rot=[100,50,0], title='removal', half_sky=True, min=-300, max=300)
    # hp.orthview(m_inpaint, rot=[100,50,0], title='inpaint', half_sky=True, min=-300, max=300)
    # plt.show()

    flux_idx = 1
    lon = np.rad2deg(df.at[flux_idx, 'lon'])
    lat = np.rad2deg(df.at[flux_idx, 'lat'])

    hp.gnomview(m_removal, rot=[lon, lat, 0], title=f'method: removal, {flux_idx=}', xsize=fig_size,ysize=fig_size)
    hp.gnomview(m_inpaint, rot=[lon, lat, 0], title=f'method: inpaint, {flux_idx=}', xsize=fig_size,ysize=fig_size)
    hp.gnomview(m_origin, rot=[lon, lat, 0], title=f'ps + cmb + noise, {flux_idx=}', xsize=fig_size,ysize=fig_size)
    hp.gnomview(m_noise, rot=[lon, lat, 0], title=f'cmb + noise, {flux_idx=}', xsize=fig_size,ysize=fig_size)
    hp.gnomview(m_removal_res, rot=[lon, lat, 0], title=f'removal residual, {flux_idx=}', xsize=fig_size,ysize=fig_size)
    hp.gnomview(m_inpaint_res, rot=[lon, lat, 0], title=f'inpaint residual, {flux_idx=}', xsize=fig_size,ysize=fig_size)
    plt.show()


def cpr_ps_cmb_fg_noise(df):
    m_res = np.load('./ps_cmb_fg_noise_residual/2sigma/map0.npy')
    mask_list = np.load('./ps_cmb_noise_residual/2sigma/mask0.npy')
    print(f'{mask_list=}')

    m_noise = np.load(f'../../../../fitdata/synthesis_data/2048/CMBFGNOISE/{freq}/0.npy')[0].copy()
    m_removal = m_res + m_noise

    m_inpaint = hp.read_map('./INPAINT/output/pcfn/2sigma/0.fits', field=0)
    m_origin = np.load(f'../../../../fitdata/synthesis_data/2048/PSCMBFGNOISE/{freq}/0.npy')[0].copy()
    m_inpaint_res = m_inpaint - m_noise
    m_removal_res = m_removal - m_noise

    # hp.orthview(m_removal, rot=[100,50,0], title='removal', half_sky=True, min=-300, max=300)
    # hp.orthview(m_inpaint, rot=[100,50,0], title='inpaint', half_sky=True, min=-300, max=300)
    # plt.show()

    flux_idx = 1
    lon = np.rad2deg(df.at[flux_idx, 'lon'])
    lat = np.rad2deg(df.at[flux_idx, 'lat'])

    hp.gnomview(m_removal, rot=[lon, lat, 0], title=f'method: removal, {flux_idx=}', xsize=fig_size,ysize=fig_size)
    hp.gnomview(m_inpaint, rot=[lon, lat, 0], title=f'method: inpaint, {flux_idx=}', xsize=fig_size,ysize=fig_size)
    hp.gnomview(m_origin, rot=[lon, lat, 0], title=f'ps + cmb + fg + noise, {flux_idx=}', xsize=fig_size,ysize=fig_size)
    hp.gnomview(m_noise, rot=[lon, lat, 0], title=f'fg + cmb + noise, {flux_idx=}', xsize=fig_size,ysize=fig_size)
    hp.gnomview(m_removal_res, rot=[lon, lat, 0], title=f'removal residual, {flux_idx=}', xsize=fig_size,ysize=fig_size)
    hp.gnomview(m_inpaint_res, rot=[lon, lat, 0], title=f'inpaint residual, {flux_idx=}', xsize=fig_size,ysize=fig_size)
    plt.show()




if __name__ == '__main__':
    df = pd.read_csv(f'../../../mask/mask_csv/{freq}.csv')
    # cpr_ps_cmb_noise(df=df)
    cpr_ps_cmb_fg_noise(df=df)




