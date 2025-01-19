import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
import os,sys

from pathlib import Path
from config import lmax, nside, freq, beam, ps_number
from eblc_base_slope import EBLeakageCorrection

def gen_map(beam, freq, lmax, rlz_idx=0, mode='mean', return_noise=False):
    # mode can be mean or std
    noise_seed = np.load('../seeds_noise_2k.npy')
    cmb_seed = np.load('../seeds_cmb_2k.npy')
    nside = 2048

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

    m = noise + ps + cmb_iqu + fg
    return m

def gen_cmb(beam, freq, lmax, rlz_idx=0, mode='mean'):
    # mode can be mean or std
    cmb_seed = np.load('../seeds_cmb_2k.npy')

    cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    if mode=='std':
        np.random.seed(seed=cmb_seed[rlz_idx])
    elif mode=='mean':
        np.random.seed(seed=cmb_seed[0])

    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=3*nside-1)

    m = cmb_iqu
    return m

def get_fiducial_factor():
    cmb = gen_cmb(beam=beam, freq=freq, lmax=lmax)
    pcfn = gen_map(beam=beam, freq=freq, lmax=lmax)

    mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/BIN_C1_5APO_3.npy')

    print(f'cmb slope is:')
    obj_cmb = EBLeakageCorrection(m=cmb, lmax=lmax, nside=nside, mask=mask, post_mask=mask)
    obj_cmb.run_eblc()
    slope_cmb = obj_cmb.return_slope()
    print(f'pcfn slope is:')
    obj_pcfn = EBLeakageCorrection(m=pcfn, lmax=lmax, nside=nside, mask=mask, post_mask=mask)
    obj_pcfn.run_eblc()
    slope_pcfn = obj_pcfn.return_slope()

    Path('test/slope').mkdir(exist_ok=True, parents=True)

    np.save('./test/slope/cmb.npy', slope_cmb)
    np.save('./test/slope/pcfn.npy', slope_pcfn)

def filterHealpyMap(hp_map: np.ndarray, lmin: int, lmax: int, nside: int) -> np.ndarray:

    # Convert map to spherical harmonics
    alms = hp.map2alm(hp_map, lmax=lmax)

    # Define filter function f_l
    fl = np.zeros(lmax + 1)  # Initialize filter with zeros
    fl[lmin:lmax + 1] = 1  # Allow only the selected l-range
    plt.loglog(fl)
    plt.show()

    # Apply filter to alm coefficients
    alm_filter = [hp.almxfl(alm, fl, inplace=True) for alm in alms]

    # Convert back to real-space map
    filtered_map = hp.alm2map(alm_filter, nside=nside, lmax=lmax)

    return filtered_map

def try_filter_alm(lmin):
    mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/BIN_C1_5APO_3.npy')

    pcfn = gen_map(beam=beam, freq=freq, lmax=lmax)
    pcfn_filtered = filterHealpyMap(hp_map=pcfn, lmin=lmin, lmax=lmax, nside=nside)

    obj_eblc = EBLeakageCorrection(m=pcfn_filtered, lmax=lmax, nside=nside, mask=mask, post_mask=mask)
    obj_eblc.run_eblc()
    slope = obj_eblc.return_slope()
    np.save(f'./test/slope/{lmin}.npy', slope)
    # hp.orthview(pcfn[1]*mask, rot=[100,50,0], title='pcfn')
    # hp.orthview(pcfn_filtered[1]*mask, rot=[100,50,0], title='pcfn filtered_map')
    # plt.show()

def check_res():
    cmb_slope = np.load('./test/slope/cmb.npy')
    pcfn_slope = np.load('./test/slope/pcfn.npy')
    print(f'{cmb_slope=}, {pcfn_slope=}')

    for lmin in [30,80,120,150,210,300]:
        slope = np.load(f'./test/slope/{lmin}.npy')
        # slope = np.load(f'./test/slope/')
        print(f'{lmin}, {slope=}')



# get_fiducial_factor()
# lmin = 120
# try_filter_alm(lmin=lmin)
check_res()




