import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
import os,sys
import logging
import ipdb
import gc

from pathlib import Path
from iminuit import Minuit
from iminuit.cost import LeastSquares

from fit_qu_base import FitPolPS

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)

freq = 270
lmax = 1999
nside = 2048
beam = 9

flux_idx = 1

def gen_c_rlz():
    cl = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')
    logger.debug(f'{cl.shape=}')
    logger.debug(f'{cl[0:10,0]=}')
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)
    logger.debug(f'{bl.shape=}')
    logger.debug(f'{bl[0:10,1]=}')

    add_beam_cl = cl * bl**2
    m = hp.synfast(cls=cl.T, nside=nside, lmax=lmax, new=True, fwhm=np.deg2rad(beam) / 60)

    ### for testing the map's power spectrum
    # l = np.arange(lmax+1)
    # dl_factor = l * (l+1) / (2 * np.pi)

    # plt.plot(cl[:,0] * dl_factor, label='no beam TT')
    # plt.plot(cl[:,1] * dl_factor, label='no beam EE')
    # plt.plot(cl[:,2] * dl_factor, label='no beam BB')
    # plt.plot(add_beam_cl[:,0] * dl_factor, label='with beam TT')
    # plt.plot(add_beam_cl[:,1] * dl_factor, label='with beam EE')
    # plt.plot(add_beam_cl[:,2] * dl_factor, label='with beam BB')
    # plt.legend()
    # plt.semilogy()
    # plt.show()

    # m_wrong = np.load('../../fitdata/2048/CMB/155/0.npy')
    # cl_wrong = hp.anafast(m_fuck, lmax=2000)

    # m = hp.synfast(cls=cl.T, nside=nside, lmax=1999, new=True, fwhm=np.deg2rad(beam) / 60)

    # cl_exp = hp.anafast(m, lmax=2000)
    # logger.debug(f'{cl_exp.shape=}')

    # # plt.plot(add_beam_cl[:,0] * dl_factor, label='with beam TT')
    # plt.plot(cl_exp[2,:] * dl_factor, label='exp TT')
    # plt.plot(cl_fuck[2,:] * dl_factor, label='fuck TT')
    # plt.legend()
    # plt.show()

    return m

def coadd_pcn(ps, nstd):
    cmb = gen_c_rlz()
    noise = nstd * np.random.normal(0, 1, size=(3, hp.nside2npix(nside=nside)))

    pcn = cmb + ps + noise
    return pcn


def main():
    time0 = time.perf_counter()
    # m = np.load(f'../../fitdata/synthesis_data/2048/PSNOISE/{freq}/0.npy')
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    ps = np.load(f'../../fitdata/2048/PS/{freq}/ps.npy')
    # m = np.load(f'../../fitdata/synthesis_data/2048/PSCMBNOISE/{freq}/2.npy')

    logger.info(f'time for fitting = {time.perf_counter()-time0}')
    nstd_q = nstd[1].copy()
    nstd_u = nstd[2].copy()
    df_mask = pd.read_csv(f'../mask/mask_csv/{freq}.csv')
    df_ps = pd.read_csv(f'../mask/ps_csv/{freq}.csv')


    fit_q_amp_list = []
    fit_u_amp_list = []
    for rlz_idx in range(100):
        m = coadd_pcn(ps=ps, nstd=nstd)
        m_q = m[1].copy()
        m_u = m[2].copy()
        logger.debug(f'{sys.getrefcount(m_q)-1=}')

        obj = FitPolPS(m_q=m_q, m_u=m_u, freq=freq, nstd_q=nstd_q, nstd_u=nstd_u, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=0.00001)

        logger.debug(f'{sys.getrefcount(m_q)-1=}')
        # obj.see_true_map(m_q=m_q, m_u=m_u, nside=nside, beam=beam)

        num_ps, chi2dof, fit_q_amp, fit_q_amp_err, fit_u_amp, fit_u_amp_err, fit_error_q, fit_error_u = obj.fit_all(cov_mode='cmb+noise')

        del m
        del obj

        gc.collect()

        fit_q_amp_list.append(fit_q_amp)
        fit_u_amp_list.append(fit_u_amp)

    path_check_bias = Path(f'./fit_res/2048/PSCMBNOISE/check_bias/idx_{flux_idx}')
    path_check_bias.mkdir(parents=True, exist_ok=True)
    np.save(path_check_bias / Path('./q_amp.npy'), np.array(fit_q_amp_list))
    np.save(path_check_bias / Path('./u_amp.npy'), np.array(fit_u_amp_list))


if __name__ == '__main__':
    # gen_pcn_rlz()
    main()



