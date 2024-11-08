import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from config import freq, lmax, beam, nside

noise_seed = np.load('../seeds_noise_2k.npy')
cmb_seed = np.load('../seeds_cmb_2k.npy')
fg_seed = np.load('../seeds_fg_2k.npy')

rlz_idx=0
nside = 2048

def gen_fg_cl():
    cl_fg = np.load('./data/debeam_full_b/cl_fg.npy')
    Cl_TT = cl_fg[0]
    Cl_EE = cl_fg[1]
    Cl_BB = cl_fg[2]
    Cl_TE = np.zeros_like(Cl_TT)
    return np.array([Cl_TT, Cl_EE, Cl_BB, Cl_TE])

def gen_map(beam, freq, lmax):
    ps = np.load('./data/ps/ps.npy')
    # fg = np.load(f'../../fitdata/2048/FG/{freq}/fg.npy')

    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    npix = hp.nside2npix(nside=2048)
    np.random.seed(seed=noise_seed[rlz_idx])
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    print(f"{np.std(noise[1])=}")

    cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    np.random.seed(seed=cmb_seed[rlz_idx])
    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=lmax)

    cls_fg = gen_fg_cl()
    np.random.seed(seed=fg_seed[rlz_idx])
    fg = hp.synfast(cls_fg, nside=nside, fwhm=0, new=True, lmax=lmax)

    # l = np.arange(lmax+1)
    # cls_out = hp.anafast(cmb_iqu, lmax=lmax)

    m = noise + ps + cmb_iqu + fg
    # m = noise

    path_fg = Path(f'./data_maps/fg')
    path_cmb = Path(f'./data_maps/cmb')
    path_noise = Path(f'./data_maps/noise')
    path_pcfn = Path(f'./data_maps/pcfn')
    path_fg.mkdir(exist_ok=True, parents=True)
    path_cmb.mkdir(exist_ok=True, parents=True)
    path_noise.mkdir(exist_ok=True, parents=True)
    path_pcfn.mkdir(exist_ok=True, parents=True)

    np.save(path_fg / Path(f'{rlz_idx}.npy'), fg)
    np.save(path_cmb / Path(f'{rlz_idx}.npy'), cmb_iqu)
    np.save(path_noise / Path(f'{rlz_idx}.npy'), noise)
    np.save(path_pcfn / Path(f'{rlz_idx}.npy'), m)

    return m

def check_map():
    df = pd.read_csv(f'./mask/{freq}.csv')
    # pcfn = gen_map(beam=beam, freq=freq, lmax=lmax)
    pcfn = np.load(f'./data_maps/pcfn/{rlz_idx}.npy')
    # ps = np.load('./data/ps/ps.npy')
    pcfn_rmv = np.load(f'./fit_res/pcfn_fit_qu/3sigma/map_q_{rlz_idx}.npy')
    bin_mask = hp.read_map('./inpainting/mask/mask.fits')
    apo_mask = np.load('./inpainting/mask/apo_ps_mask.npy')
    lon = np.rad2deg(df.at[rlz_idx, 'lon'])
    lat = np.rad2deg(df.at[rlz_idx, 'lat'])
    hp.gnomview(pcfn[1], rot=[lon,lat,0], title='pcfn')
    hp.gnomview(pcfn_rmv, rot=[lon,lat,0], title='rmv')
    hp.gnomview(pcfn[1]-pcfn_rmv, rot=[lon,lat,0], title='pcfn - rmv')
    hp.orthview(bin_mask, rot=[100,50,0], title='bin_mask')
    hp.orthview(apo_mask, rot=[100,50,0], title='apo_mask')
    plt.show()


check_map()




