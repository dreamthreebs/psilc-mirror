import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os,sys
import pandas as pd

from pathlib import Path
from eblc_base import EBLeakageCorrection

lmax = 1999
nside = 2048
threshold = 2
mask = np.load('../../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
rlz_idx=78

def test_eblc():
    rlz_idx = 0
    q = np.load(f'./{threshold}sigma/map_q_{rlz_idx}.npy')
    u = np.load(f'./{threshold}sigma/map_u_{rlz_idx}.npy')
    i = np.zeros_like(q)
    m = np.array([i,q,u])
    print(f'{m.shape=}')

    obj_eblc = EBLeakageCorrection(m, lmax=lmax, nside=nside, mask=mask, post_mask=mask, check_res=True)
    crt_b, tmp_b, cln_b = obj_eblc.run_eblc()

def run_eblc():
    q = np.load(f'./{threshold}sigma/map_q_{rlz_idx}.npy')
    u = np.load(f'./{threshold}sigma/map_u_{rlz_idx}.npy')
    i = np.zeros_like(q)
    m = np.array([i,q,u])
    print(f'{m.shape=}')

    obj_eblc = EBLeakageCorrection(m, lmax=lmax, nside=nside, mask=mask, post_mask=mask, check_res=False)
    crt_b, tmp_b, cln_b = obj_eblc.run_eblc()
    path_eblc_b = Path(f'./{threshold}sigma/B')
    path_eblc_b.mkdir(exist_ok=True, parents=True)
    np.save(path_eblc_b / Path(f'map_cln_b{rlz_idx}.npy'), cln_b)

def from_qu_to_E():
    print(f'{rlz_idx=}')
    q = np.load(f'./{threshold}sigma/map_q_{rlz_idx}.npy')
    u = np.load(f'./{threshold}sigma/map_u_{rlz_idx}.npy')
    i = np.zeros_like(q)
    m = np.array([i,q,u])
    print(f'{m.shape=}')

    crp_e = hp.alm2map(hp.map2alm(m * mask, lmax=lmax)[1], nside=nside) * mask
    path_eblc_e = Path(f'./{threshold}sigma/E')
    path_eblc_e.mkdir(exist_ok=True, parents=True)
    np.save(path_eblc_e / Path(f'map_crp_e{rlz_idx}.npy'), crp_e)

    # m_cmb = np.load('../../../../../fitdata/synthesis_data/2048/CMBNOISE/270/1.npy')
    # pure_e = hp.alm2map(hp.map2alm(m_cmb, lmax=lmax)[1], nside=nside) * mask

    # hp.orthview(crp_e, rot=[100,50,0], half_sky=True, title='corrupted E')
    # hp.orthview(pure_e, rot=[100,50,0], half_sky=True, title='pure E')
    # hp.orthview(crp_e - pure_e, rot=[100,50,0], half_sky=True, title='residual')
    # plt.show()


# run_eblc()
from_qu_to_E()




