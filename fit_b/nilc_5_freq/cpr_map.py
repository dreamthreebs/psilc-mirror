import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../30GHz/mask/30_after_filter.csv')

flux_idx = 2
rlz_idx = 0

def plot_map():

    lon = np.rad2deg(df.at[flux_idx, 'lon'])
    lat = np.rad2deg(df.at[flux_idx, 'lat'])
    print(f'{lon=}, {lat=}')
    m_pcfn = np.load(f'./data2/std/pcfn/{rlz_idx}.npy')

    fig = plt.figure(figsize=(12,8))
    vmin = -0.8
    vmax = 0.8

    hp.gnomview(m_pcfn, rot=[lon, lat, 0], sub=(141), notext=True, cbar=False, title='PS + CMB + FG + NOISE', xsize=120, min=vmin, max=vmax)

    m_cfn = np.load(f'./data2/std/cfn/{rlz_idx}.npy')
    hp.gnomview(m_cfn, rot=[lon, lat, 0], sub=(142), notext=True, cbar=False, title='CMB + FG + NOISE', xsize=120, min=vmin, max=vmax)

    m_rmv = np.load(f'./data2/std/rmv/{rlz_idx}.npy')
    hp.gnomview(m_rmv, rot=[lon, lat, 0], sub=(143), notext=True, cbar=False, title='Template fitting method', xsize=120, min=vmin, max=vmax)

    m_inp = np.load(f'./data2/std/inp/{rlz_idx}.npy')
    hp.gnomview(m_inp, rot=[lon, lat, 0], sub=(144), notext=True, cbar=False, title='Recycling + Inpainting on B', xsize=120, min=vmin, max=vmax)

    # Find global min and max
    # vmin = min(m_pcfn.min(), m_cfn.min())
    # vmax = max(m_pcfn.max(), m_cfn.max())


    # Add a shared colorbar
    cax = fig.add_axes([0.35, 0.23, 0.3, 0.03])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('$\\mu KCMB$')

    # plt.tight_layout()
    plt.subplots_adjust()
    plt.savefig('./pcfn.pdf', bbox_inches='tight')
    # plt.show()

plot_map()


