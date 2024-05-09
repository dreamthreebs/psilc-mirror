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
rlz_idx=0

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

run_eblc()


