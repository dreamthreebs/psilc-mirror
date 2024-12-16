import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
import os,sys
import logging
import ipdb

from pathlib import Path
from iminuit import Minuit
from iminuit.cost import LeastSquares
from fit_qu_no_const import FitPolPS

from config import freq, lmax, beam, nside
print(f'{freq=}, {lmax=}, {beam=}, {nside=}')

# logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s -%(name)s - %(message)s')
logging.basicConfig(level=logging.WARNING)
# logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)

def main():
    nside = 1024
    nstd = np.load(f'../../FGSim/NSTDNORTH/1024/{freq}.npy')
    nstd_q = nstd[1].copy()
    nstd_u = nstd[2].copy()

    time0 = time.perf_counter()
    # m = np.load(f'../../fitdata/synthesis_data/2048/PSNOISE/{freq}/0.npy')
    # m = np.load(f'../../fitdata/synthesis_data/2048/PSCMBNOISE/{freq}/3.npy')
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3,hp.nside2npix(nside)))
    m = noise
    m_q = m[1].copy()
    m_u = m[2].copy()
    logger.debug(f'{sys.getrefcount(m_q)-1=}')

    logger.info(f'time for fitting = {time.perf_counter()-time0}')

    df_mask = pd.read_csv(f'./mask/{freq}.csv')
    print(f'{len(df_mask)=}')
    df_ps = df_mask

    for flux_idx in range(len(df_mask)):
        flux_idx = 0

        logger.debug(f'{sys.getrefcount(m_q)-1=}')
        obj = FitPolPS(m_q=m_q, m_u=m_u, freq=freq, nstd_q=nstd_q, nstd_u=nstd_u, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_mask, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=0.00001)
        # num_ps, num_near_ps, ang_near = obj.fit_all(mode='get_num_ps', cov_mode='cmb+noise')
        # logger.info(f'{num_ps=}, {num_near_ps}, {ang_near=}')

        logger.debug(f'{sys.getrefcount(m_q)-1=}')
        # obj.calc_definite_fixed_cmb_cov()
        # obj.calc_covariance_matrix(mode='cmb+noise')
        # obj.fit_all(cov_mode='cmb+noise')

        # obj.calc_covariance_matrix(mode='noise')
        # obj.fit_all(cov_mode='noise')

        # obj.fit_all(cov_mode='noise', mode='check_sigma')

main()




