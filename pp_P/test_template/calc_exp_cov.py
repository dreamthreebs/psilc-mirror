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
from numpy.polynomial.legendre import Legendre
from scipy.interpolate import CubicSpline
from memory_profiler import profile

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from fit_qu_base import FitPolPS

def main():
    freq =155
    time0 = time.perf_counter()
    i_list = []
    q_list = []
    u_list = []

    idx = np.load('./ipix_fit_qu.npy')
    for rlz_idx in range(1000):
        print(f'{rlz_idx=}')
        m = np.load(f'../../fitdata/2048/CMB/{freq}/{rlz_idx}.npy').copy()
        m_i = m[0][idx].copy()
        m_q = m[1][idx].copy()
        m_u = m[2][idx].copy()

        i_list.append(m_i)
        q_list.append(m_q)
        u_list.append(m_u)

    i_arr = np.array(i_list)
    q_arr = np.array(q_list)
    u_arr = np.array(u_list)
    print(f'{q_arr.shape=}')

    qu_arr = np.concatenate([q_arr, u_arr], axis=1)
    cov_i = np.cov(i_arr, rowvar=False)
    cov_qu = np.cov(qu_arr, rowvar=False)
    print(f'{cov_i=}')
    print(f'{cov_qu=}')

    np.save(f'exp_cov_I.npy', cov_i)
    np.save(f'exp_cov_QU.npy', cov_qu)

main()


