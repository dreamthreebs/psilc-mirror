import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
import os,sys
import logging

from pathlib import Path
from config import lmax, nside, freq, beam, ps_number
from eblc_base_slope import EBLeakageCorrection

noise_seeds = np.load('../seeds_noise_2k.npy')
cmb_seeds = np.load('../seeds_cmb_2k.npy')

def gen_map(beam, freq, rlz_idx=0, mode='mean', return_noise=False):
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

def gen_cmb(beam, freq, rlz_idx=0, mode='mean', return_noise=False):
    # mode can be mean or std
    noise_seed = np.load('../seeds_noise_2k.npy')
    cmb_seed = np.load('../seeds_cmb_2k.npy')
    nside = 2048


    cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    if mode=='std':
        np.random.seed(seed=cmb_seed[rlz_idx])
    elif mode=='mean':
        np.random.seed(seed=cmb_seed[0])

    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=3*nside-1)

    m = cmb_iqu
    return m


def smooth(map_in, lmax, beam_in, beam_out):
    # map_in should be in (3,npix)

    bl_in = hp.gauss_beam(fwhm=np.deg2rad(beam_in)/60, lmax=lmax, pol=True) # (lmax+1,4)
    bl_out = hp.gauss_beam(fwhm=np.deg2rad(beam_out)/60, lmax=lmax, pol=True)
    print(f'{bl_in.shape=}')
    alms = hp.map2alm(map_in, lmax)
    sm_alm = np.asarray([hp.almxfl(alm, bl_out[:,i]/bl_in[:,i]) for i, alm in enumerate(alms)])
    print(f'{sm_alm.shape=}')

    map_out = hp.alm2map(sm_alm, nside=nside)
    return map_out

def try_smooth():
    beam_base = 17 # arcmin
    mask = np.load(f'../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_3.npy')
    mask_for_cl = np.load(f'../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_3APO_5.npy')
    mask_for_eblc = np.load(f'../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/BIN_C1_5APO_3.npy')
    # m_67 = gen_cmb(beam=beam, freq=freq)
    # m_17 = gen_cmb(beam=beam_base, freq=freq)
    # m_67to17 = smooth(map_in=m_67*mask, lmax=lmax, beam_in=beam, beam_out=beam_base)

    # hp.mollview(m_17[0]*mask_for_cl)
    # hp.mollview(m_17[1]*mask_for_cl)
    # hp.mollview(m_17[2]*mask_for_cl)
    # hp.mollview(map_out[0]*mask_for_cl)
    # hp.mollview(map_out[1]*mask_for_cl)
    # hp.mollview(map_out[2]*mask_for_cl)
    # plt.show()

    # cl_17 = hp.anafast(m_17*mask_for_cl, lmax=lmax)
    # cl_67to17 = hp.anafast(map_out*mask_for_cl, lmax=lmax)
    # np.save(f'./test/cl_17.npy', cl_17)
    # np.save(f'./test/cl_67to17.npy', cl_67to17)

    # cl_17 = np.load(f'./test/cl_17.npy')
    # cl_67to17 = np.load(f'./test/cl_67to17.npy')

    # plt.loglog(cl_17[2], label='17 arcmin')
    # plt.loglog(cl_67to17[2], label='67 to 17 arcmin')
    # plt.legend()
    # plt.show()

    # obj_67to17 = EBLeakageCorrection(m=m_67to17, lmax=lmax, nside=nside, mask=mask_for_eblc, post_mask=mask_for_eblc)
    # _, _, cln_67to17 = obj_67to17.run_eblc()

    # obj_17 = EBLeakageCorrection(m=m_17, lmax=lmax, nside=nside, mask=mask_for_eblc, post_mask=mask_for_eblc)
    # _, _, cln_17 = obj_17.run_eblc()

    # np.save(f'./test/cln_67to17.npy', cln_67to17)
    # np.save(f'./test/cln_17.npy', cln_17)

    cln_67to17 = np.load(f'./test/cln_67to17.npy')
    cln_17 = np.load(f'./test/cln_17.npy')

    hp.orthview(cln_67to17*mask_for_cl, rot=[100,50,0], title='67to17')
    hp.orthview(cln_17*mask_for_cl, rot=[100,50,0], title='17')
    plt.show()

    cl_17 = hp.anafast(cln_17 * mask_for_cl, lmax=lmax)
    cl_67to17 = hp.anafast(cln_67to17 * mask_for_cl, lmax=lmax)
    plt.loglog(cl_17, label='17')
    plt.loglog(cl_67to17, label='67 to 17')
    plt.show()


try_smooth()


