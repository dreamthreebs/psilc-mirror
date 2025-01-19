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

cmb = gen_cmb(beam=beam, freq=freq, lmax=lmax)
pcfn = gen_map(beam=beam, freq=freq, lmax=lmax)

mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/BIN_C1_5APO_3.npy')
print(f'cmb slope is:')
obj_cmb = EBLeakageCorrection(m=cmb, lmax=lmax, nside=nside, mask=mask, post_mask=mask)
obj_cmb.run_eblc()
print(f'pcfn slope is:')
obj_pcfn = EBLeakageCorrection(m=pcfn, lmax=lmax, nside=nside, mask=mask, post_mask=mask)
obj_pcfn.run_eblc()

