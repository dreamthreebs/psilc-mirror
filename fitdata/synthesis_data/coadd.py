import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os

nside = 256
freq = 40

for n_rlz in range(100):
    print(f'{n_rlz=}')

    cmb = np.load(f'../{nside}/CMB/{freq}/{n_rlz}.npy')
    fg = np.load(f'../{nside}/FG/{freq}/fg.npy')
    ps = np.load(f'../{nside}/PS/{freq}/ps.npy')
    noise = np.load(f'../{nside}/NOISE/{freq}/{n_rlz}.npy')
    
    pscmbfgnoise = ps + cmb + fg + noise
    cmbfgnoise = cmb + fg + noise

    pscmbnoise = ps + cmb + noise
    cmbnoise = cmb + noise

    psnoise = ps + noise

    os.makedirs(f'./{nside}/PSCMBFGNOISE/{freq}', exist_ok=True)
    np.save(f'./{nside}/PSCMBFGNOISE/{freq}/{n_rlz}.npy', pscmbfgnoise)

    os.makedirs(f'./{nside}/CMBFGNOISE/{freq}', exist_ok=True)
    np.save(f'./{nside}/CMBFGNOISE/{freq}/{n_rlz}.npy', cmbfgnoise)

    os.makedirs(f'./{nside}/PSCMBNOISE/{freq}', exist_ok=True)
    np.save(f'./{nside}/PSCMBNOISE/{freq}/{n_rlz}.npy', pscmbnoise)

    os.makedirs(f'./{nside}/CMBNOISE/{freq}', exist_ok=True)
    np.save(f'./{nside}/CMBNOISE/{freq}/{n_rlz}.npy', cmbnoise)

    os.makedirs(f'./{nside}/PSNOISE/{freq}', exist_ok=True)
    np.save(f'./{nside}/PSNOISE/{freq}/{n_rlz}.npy', psnoise)

