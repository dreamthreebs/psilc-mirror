import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from config import freq, nside, beam
from eblc_base_slope import EBLeakageCorrection

lmax = 1500
rlz_idx = 0

mask = np.load('../../src/mask/north/BINMASKG2048.npy')

noise_seed = np.load('../seeds_noise_2k.npy')
cmb_seed = np.load('../seeds_cmb_2k.npy')
fg_seed = np.load('../seeds_fg_2k.npy')

flux_idx = 1

def gen_cmb_b():
    beam_base = 17
    cmb_seed = np.load('../seeds_cmb_2k.npy')
    cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')

    np.random.seed(seed=cmb_seed[0])
    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam_base)/60, new=True, lmax=3*nside-1)
    cmb_b = hp.alm2map(hp.map2alm(cmb_iqu, lmax=lmax)[2], nside=nside)
    np.save(f'./paper/B_map_1500/cmb_b_{rlz_idx}.npy', cmb_b)
    return cmb_b


def gen_map(rlz_idx=0, mode='mean', return_noise=False):
    # mode can be mean or std

    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    npix = hp.nside2npix(nside=2048)
    np.random.seed(seed=noise_seed[rlz_idx])
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    print(f"{np.std(noise[1])=}")

    if return_noise:
        return noise

    ps = np.load(f'../../fitdata/2048/PS/{freq}/ps.npy')
    fg = np.load(f'../../fitdata/2048/FG/{freq}/fg.npy')

    cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    if mode=='std':
        np.random.seed(seed=cmb_seed[rlz_idx])
    elif mode=='mean':
        np.random.seed(seed=cmb_seed[0])

    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=3*nside-1)

    pcfn = noise + ps + cmb_iqu + fg
    cfn = noise + cmb_iqu + fg
    cf = cmb_iqu + fg
    n = noise
    return pcfn, cfn, cf, n

def gen_ps(rlz_idx=0):
    # mode can be mean or std

    ps = np.load(f'../../fitdata/2048/PS/{freq}/ps.npy')

    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    npix = hp.nside2npix(nside=2048)
    np.random.seed(seed=noise_seed[rlz_idx])
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    print(f"{np.std(noise[1])=}")

    ps = ps + noise

    return ps


def convert_to_B():

    # generate map: pcfn, cfn, rmv, inp
    m_pcfn, m_cfn, m_cf, m_n= gen_map(rlz_idx=rlz_idx, mode='std')

    m_rmv_q = np.load(f'./fit_res/std/3sigma/map_q_{rlz_idx}.npy')
    m_rmv_u = np.load(f'./fit_res/std/3sigma/map_u_{rlz_idx}.npy')
    m_rmv = np.asarray([np.zeros_like(m_rmv_q), m_rmv_q, m_rmv_u])

    obj_pcfn = EBLeakageCorrection(m_pcfn, lmax=lmax, nside=nside, mask=mask, post_mask=mask)
    _,_,cln_b_pcfn = obj_pcfn.run_eblc()

    obj_cfn = EBLeakageCorrection(m_cfn, lmax=lmax, nside=nside, mask=mask, post_mask=mask)
    _,_,cln_b_cfn = obj_cfn.run_eblc()

    obj_rmv = EBLeakageCorrection(m_rmv, lmax=lmax, nside=nside, mask=mask, post_mask=mask)
    _,_,cln_b_rmv = obj_rmv.run_eblc()

    path_map = Path('./paper/B_map_1500')
    path_map.mkdir(exist_ok=True, parents=True)

    np.save(path_map / Path(f'pcfn_{rlz_idx}.npy'), cln_b_pcfn)
    np.save(path_map / Path(f'cfn_{rlz_idx}.npy'), cln_b_cfn)
    np.save(path_map / Path(f'rmv_{rlz_idx}.npy'), cln_b_rmv)

def plot_each_freq_map():
    m_pcfn = np.load(f'./paper/B_map/pcfn_{rlz_idx}.npy')
    m_cfn = np.load(f'./paper/B_map/cfn_{rlz_idx}.npy')
    m_rmv = np.load(f'./paper/B_map/rmv_{rlz_idx}.npy')
    m_inp = hp.read_map(f'./inpainting/output_m3_std_new/{rlz_idx}.fits')
    # m_cmb_b = np.load(f'./paper/B_map_1500/cmb_b_0.npy')

    fig = plt.figure(figsize=(12,8))

    df = pd.read_csv('./mask/155_after_filter.csv')
    lon = np.rad2deg(df.at[flux_idx, 'lon'])
    lat = np.rad2deg(df.at[flux_idx, 'lat'])

    vmin = -0.8
    vmax = 0.8

    hp.gnomview(m_cfn, rot=[lon, lat, 0], sub=(141), notext=True, cbar=False, title='Simulation without PS', xsize=120, min=vmin, max=vmax)
    hp.gnomview(m_pcfn, rot=[lon, lat, 0], sub=(142), notext=True, cbar=False, title='Simulation with PS', xsize=120, min=vmin, max=vmax)
    # hp.gnomview(m_cmb_b, rot=[lon, lat, 0], sub=(153), notext=True, cbar=False, title='Fiducial CMB', xsize=120, min=vmin, max=vmax)
    hp.gnomview(m_rmv, rot=[lon, lat, 0], sub=(143), notext=True, cbar=False, title='GLSPF', xsize=120, min=vmin, max=vmax)
    hp.gnomview(m_inp, rot=[lon, lat, 0], sub=(144), notext=True, cbar=False, title='Inpainting', xsize=120, min=vmin, max=vmax)

    # 2) now grab the first axes (it's the first one created)
    ax1 = fig.axes[0]

    # 3) shift it left by 0.02 in figure‐fraction coordinates
    # x0, y0, w, h = ax1.get_position().bounds
    # ax1.set_position([x0 - 0.02, y0, w, h])

    # 4) draw your vertical dashed line
    ax1.plot([1.02, 1.02], [0, 1],
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
    plt.savefig(f'/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20250609/{freq}.pdf', bbox_inches='tight')
    # plt.show()
    plt.show()

def plot_QU():

    # m_pcfn, m_cfn, m_cf, m_n= gen_map(rlz_idx=rlz_idx, mode='std')
    m_p = gen_ps()

    df = pd.read_csv('./mask/155_after_filter.csv')
    lon = np.rad2deg(df.at[flux_idx, 'lon'])
    lat = np.rad2deg(df.at[flux_idx, 'lat'])

    # hp.gnomview(m_pcfn[1], rot=[lon, lat, 0])
    # hp.gnomview(m_cfn[1], rot=[lon, lat, 0])
    # plt.show()

    hp.gnomview(m_p[1], rot=[lon, lat, 0])
    plt.show()


# convert_to_B()
# gen_cmb_b()
plot_each_freq_map()
# plot_QU()






