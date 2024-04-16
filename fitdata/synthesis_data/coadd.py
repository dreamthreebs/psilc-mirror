import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os, time
from pathlib import Path

nside = 2048
freq = 270

n_rlz_begin = 1
n_rlz_end = 100

for n_rlz in range(n_rlz_begin, n_rlz_end):
    print(f'{n_rlz=}')

    time0 = time.perf_counter()
    cmb = np.load(f'../{nside}/CMB/{freq}/{n_rlz}.npy')
    fg = np.load(f'../{nside}/FG/{freq}/fg.npy')
    # ps = np.load(f'../{nside}/PS/{freq}/ps.npy')
    # noise = np.load(f'../{nside}/NOISE/{freq}/{n_rlz}.npy')
    time_load = time.perf_counter()
    print(f'load time = {time_load-time0}')
    
    # pscmbfgnoise = ps + cmb + fg + noise
    # cmbfgnoise = cmb + fg + noise

    # pscmbnoise = ps + cmb + noise
    # cmbnoise = cmb + noise

    # psnoise = ps + noise
    cmbfg = cmb + fg

    time_add = time.perf_counter()
    print(f'add time = {time_add-time_load}')

    # PSCMBFGNOISE_path = Path(f'./{nside}/PSCMBFGNOISE/{freq}')
    # PSCMBFGNOISE_path.mkdir(parents=True, exist_ok=True)
    # np.save(f'./{nside}/PSCMBFGNOISE/{freq}/{n_rlz}.npy', pscmbfgnoise)

    # CMBFGNOISE_path = Path(f'./{nside}/CMBFGNOISE/{freq}')
    # CMBFGNOISE_path.mkdir(parents=True, exist_ok=True)
    # np.save(f'./{nside}/CMBFGNOISE/{freq}/{n_rlz}.npy', cmbfgnoise)

    # PSCMBNOISE_path = Path(f'./{nside}/PSCMBNOISE/{freq}')
    # PSCMBNOISE_path.mkdir(parents=True, exist_ok=True)
    # np.save(f'./{nside}/PSCMBNOISE/{freq}/{n_rlz}.npy', pscmbnoise)

    # CMBNOISE_path = Path(f'./{nside}/CMBNOISE/{freq}')
    # CMBNOISE_path.mkdir(parents=True, exist_ok=True)
    # np.save(f'./{nside}/CMBNOISE/{freq}/{n_rlz}.npy', cmbnoise)

    # PSNOISE_path = Path(f'./{nside}/PSNOISE/{freq}')
    # PSNOISE_path.mkdir(parents=True, exist_ok=True)
    # np.save(f'./{nside}/PSNOISE/{freq}/{n_rlz}.npy', psnoise)

    CMBFG_path = Path(f'./{nside}/CMBFG/{freq}')
    CMBFG_path.mkdir(parents=True, exist_ok=True)
    np.save(f'./{nside}/CMBFG/{freq}/{n_rlz}.npy', cmbfg)

    time_save = time.perf_counter()
    print(f'save time = {time_save-time_add}')


