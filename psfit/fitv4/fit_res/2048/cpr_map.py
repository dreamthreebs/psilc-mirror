import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd

def cpr_ps_cmb_noise(df):
    m_res = np.load('./ps_cmb_noise_residual/map0.npy')
    mask_list = np.load('./ps_cmb_noise_residual/mask0.npy')
    print(f'{mask_list=}')

    m_noise = np.load('../../../../fitdata/synthesis_data/2048/CMBNOISE/40/0.npy')[0].copy()
    m_removal = m_res + m_noise

    m_inpaint = hp.read_map('./for_inpainting/output/pcn/0.fits', field=0)

    m_origin = np.load('../../../../fitdata/synthesis_data/2048/CMBNOISE/40/0.npy')[0].copy()

    # hp.orthview(m_removal, rot=[100,50,0], title='removal', half_sky=True, min=-300, max=300)
    # hp.orthview(m_inpaint, rot=[100,50,0], title='inpaint', half_sky=True, min=-300, max=300)
    # plt.show()

    flux_idx = 5
    lon = np.rad2deg(df.at[flux_idx, 'lon'])
    lat = np.rad2deg(df.at[flux_idx, 'lat'])

    hp.gnomview(m_removal, rot=[lon, lat, 0], title=f'ps removal, {flux_idx=}')
    hp.gnomview(m_inpaint, rot=[lon, lat, 0], title=f'ps inpaint, {flux_idx=}')
    hp.gnomview(m_origin, rot=[lon, lat, 0], title=f'True map, {flux_idx=}')
    plt.show()


def cpr_ps_cmb_fg_noise(df):
    m_res = np.load('./ps_cmb_fg_noise_residual/0.npy')
    mask_list = np.load('./ps_cmb_noise_residual/mask0.npy')
    print(f'{mask_list=}')

    m_noise = np.load('../../../../fitdata/synthesis_data/2048/CMBFGNOISE/40/0.npy')[0].copy()
    m_removal = m_res + m_noise

    m_inpaint = hp.read_map('./for_inpainting/output/pcfn/0.fits', field=0)

    m_origin = np.load('../../../../fitdata/synthesis_data/2048/CMBFGNOISE/40/0.npy')[0].copy()

    # hp.orthview(m_removal, rot=[100,50,0], title='removal', half_sky=True, min=-300, max=300)
    # hp.orthview(m_inpaint, rot=[100,50,0], title='inpaint', half_sky=True, min=-300, max=300)
    # plt.show()

    flux_idx = 1
    lon = np.rad2deg(df.at[flux_idx, 'lon'])
    lat = np.rad2deg(df.at[flux_idx, 'lat'])

    hp.gnomview(m_removal, rot=[lon, lat, 0], title=f'ps removal, {flux_idx=}')
    hp.gnomview(m_inpaint, rot=[lon, lat, 0], title=f'ps inpaint, {flux_idx=}')
    hp.gnomview(m_origin, rot=[lon, lat, 0], title=f'True map, {flux_idx=}')
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('../../../partial_sky_ps/ps_in_mask/2048/40mask.csv')
    # cpr_ps_cmb_noise(df=df)
    cpr_ps_cmb_fg_noise(df=df)


