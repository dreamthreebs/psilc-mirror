import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../30GHz/mask/30_after_filter.csv')

lmax = 1500
nside = 2048
flux_idx = 2
rlz_idx = 0

beam_base = 17

def gen_cmb_b():
    cmb_seed = np.load('../seeds_cmb_2k.npy')
    cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')

    np.random.seed(seed=cmb_seed[0])
    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam_base)/60, new=True, lmax=3*nside-1)
    cmb_b = hp.alm2map(hp.map2alm(cmb_iqu, lmax=lmax)[2], nside=nside)
    np.save(f'./data2/std/cmb_b_{rlz_idx}.npy', cmb_b)
    return cmb_b

def plot_map():

    lon = np.rad2deg(df.at[flux_idx, 'lon'])
    lat = np.rad2deg(df.at[flux_idx, 'lat'])
    print(f'{lon=}, {lat=}')
    m_pcfn = np.load(f'./data2/std/pcfn/{rlz_idx}.npy')
    m_cmb_b = np.load(f'./data2/std/cmb_b_{rlz_idx}.npy')
    m_cfn = np.load(f'./data2/std/cfn/{rlz_idx}.npy')

    fig = plt.figure(figsize=(12,8))
    vmin = -0.8
    vmax = 0.8

    hp.gnomview(m_cmb_b, rot=[lon, lat, 0], sub=(151), notext=True, cbar=False, title='Fiducial CMB', xsize=120, min=vmin, max=vmax)

    hp.gnomview(m_cfn, rot=[lon, lat, 0], sub=(152), notext=True, cbar=False, title='Simulation without PS', xsize=120, min=vmin, max=vmax)
    hp.gnomview(m_pcfn, rot=[lon, lat, 0], sub=(153), notext=True, cbar=False, title='Simulation with PS', xsize=120, min=vmin, max=vmax)

    m_rmv = np.load(f'./data2/std/rmv/{rlz_idx}.npy')
    hp.gnomview(m_rmv, rot=[lon, lat, 0], sub=(154), notext=True, cbar=False, title='GPSF', xsize=120, min=vmin, max=vmax)

    m_inp = np.load(f'./data2/std/inp/{rlz_idx}.npy')
    hp.gnomview(m_inp, rot=[lon, lat, 0], sub=(155), notext=True, cbar=False, title='Inpainting', xsize=120, min=vmin, max=vmax)

    # Find global min and max
    # vmin = min(m_pcfn.min(), m_cfn.min())
    # vmax = max(m_pcfn.max(), m_cfn.max())

    # 2) now grab the first axes (it's the first one created)
    ax1 = fig.axes[0]

    # 3) shift it left by 0.02 in figure‐fraction coordinates
    # x0, y0, w, h = ax1.get_position().bounds
    # ax1.set_position([x0 - 0.02, y0, w, h])

    # 4) draw your vertical dashed line
    ax1.plot([1.025, 1.025], [0, 1],
             linestyle='--', linewidth=2,
             color='black',
             transform=ax1.transAxes,
             clip_on=False)


    # Add a shared colorbar
    cax = fig.add_axes([0.35, 0.23, 0.3, 0.03])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('$\\mu KCMB$')

    # plt.tight_layout()
    plt.subplots_adjust()
    plt.savefig('/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20250726/nilc_res_map.pdf', bbox_inches='tight')
    plt.show()

# gen_cmb_b()
plot_map()


