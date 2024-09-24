import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
import os,sys,gc
import logging
import ipdb

from pathlib import Path
from iminuit import Minuit
from iminuit.cost import LeastSquares
from fit_qu_no_const import FitPolPS

# logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s -%(name)s - %(message)s')
logging.basicConfig(level=logging.WARNING)
# logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)

rlz_idx=0
nside = 2048
lmax = 3 * nside - 1
npix = hp.nside2npix(nside)
beam = 67

noise_seed = np.load('../seeds_noise_2k.npy')
cmb_seed = np.load('../seeds_cmb_2k.npy')
fg_seed = np.load('../seeds_fg_2k.npy')

def gen_fg_cl():
    Cl_TT = np.load('../Cl_fg/data/cl_fg_TT.npy')
    Cl_EE = np.load('../Cl_fg/data/cl_fg_EE.npy')
    Cl_BB = np.load('../Cl_fg/data/cl_fg_BB.npy')
    Cl_TE = np.zeros_like(Cl_TT)
    return np.array([Cl_TT, Cl_EE, Cl_BB, Cl_TE])

def gen_map():
    ps = np.load('./data/ps/ps.npy')
    # fg = np.load('../../fitdata/2048/FG/30/fg.npy')

    nstd = np.load('../../FGSim/NSTDNORTH/2048/30.npy')
    np.random.seed(seed=noise_seed[rlz_idx])
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    print(f"{np.std(noise[1])=}")

    # cmb_iqu = np.load(f'../../fitdata/2048/CMB/215/{rlz_idx}.npy')
    # cls = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')
    cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    np.random.seed(seed=cmb_seed[rlz_idx])
    # cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=1999)
    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=600)

    cls_fg = gen_fg_cl()
    np.random.seed(seed=fg_seed[rlz_idx])
    fg_iqu = hp.synfast(cls_fg, nside=nside, fwhm=0, new=True, lmax=600)

    # l = np.arange(lmax+1)
    # cls_out = hp.anafast(cmb_iqu, lmax=lmax)

    m = noise + ps + cmb_iqu + fg_iqu
    # m = noise

    # path_fg = Path(f'./data/fg')
    # path_cmb = Path(f'./data/cmb')
    # path_noise = Path(f'./data/noise')
    # path_pcfn = Path(f'./data/pcfn')
    # path_fg.mkdir(exist_ok=True, parents=True)
    # path_cmb.mkdir(exist_ok=True, parents=True)
    # path_noise.mkdir(exist_ok=True, parents=True)
    # path_pcfn.mkdir(exist_ok=True, parents=True)

    # np.save(path_fg / Path(f'{rlz_idx}.npy'), fg_iqu)
    # np.save(path_cmb / Path(f'{rlz_idx}.npy'), cmb_iqu)
    # np.save(path_noise / Path(f'{rlz_idx}.npy'), noise)
    # np.save(path_pcfn / Path(f'{rlz_idx}.npy'), m)

    return m

def main():
    freq = 30
    lmax = 1999
    nside = 2048
    beam = 67
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    nstd_q = nstd[1].copy()
    nstd_u = nstd[2].copy()

    time0 = time.perf_counter()
    # m = np.load(f'../../fitdata/synthesis_data/2048/PSNOISE/{freq}/0.npy')
    # m = np.load(f'../../fitdata/synthesis_data/2048/PSCMBNOISE/{freq}/3.npy')
    m = gen_map()

    m_q = m[1].copy()
    m_u = m[2].copy()
    logger.debug(f'{sys.getrefcount(m_q)-1=}')

    logger.info(f'time for fitting = {time.perf_counter()-time0}')

    df_mask = pd.read_csv('./mask/30.csv')
    print(f'{df_mask=}')
    df_ps = pd.read_csv('../../pp_P/mask/ps_csv/30.csv')

    logger.debug(f'{sys.getrefcount(m_q)-1=}')
    for flux_idx in range(76):
        obj = FitPolPS(m_q=m_q, m_u=m_u, freq=freq, nstd_q=nstd_q, nstd_u=nstd_u, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_mask, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=0.00001)

        logger.debug(f'{sys.getrefcount(m_q)-1=}')
        # obj.calc_definite_fixed_cmb_cov()
        # obj.calc_covariance_matrix(mode='cmb+noise')
        num_ps, chi2dof, fit_P, fit_P_err, fit_phi, fit_phi_err = obj.fit_all(cov_mode='cmb+noise')

        path_res = Path(f'./fit_res/pcfn_params/fit_qu_no_const/idx_{flux_idx}')
        path_res.mkdir(exist_ok=True, parents=True)
        print(f"{num_ps=}, {chi2dof=}, {obj.p_amp=}, {fit_P=}, {fit_P_err=}, {obj.phi=}, {fit_phi=}, {fit_phi_err=}")
        np.save(path_res / Path(f'chi2dof_{rlz_idx}.npy'), chi2dof)
        np.save(path_res / Path(f'fit_P_{rlz_idx}.npy'), fit_P)
        np.save(path_res / Path(f'P_{rlz_idx}.npy'), obj.p_amp)
        np.save(path_res / Path(f'phi_{rlz_idx}.npy'), obj.phi)
        np.save(path_res / Path(f'fit_err_P_{rlz_idx}.npy'), fit_P_err)
        np.save(path_res / Path(f'fit_phi_{rlz_idx}.npy'), fit_phi)
        np.save(path_res / Path(f'fit_err_phi_{rlz_idx}.npy'), fit_phi_err)

        del obj
        gc.collect()

main()
# gen_map()







