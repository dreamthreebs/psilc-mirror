import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from pathlib import Path

nside = 2048
lmax = 1000
beam = 67
rlz_idx=0

cmb_seeds = np.load('../seeds_cmb_2k.npy')
noise_seeds = np.load('../seeds_noise_2k.npy')

bin_mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
apo_mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')

print(f'{rlz_idx=}')
np.random.seed(seed=cmb_seeds[rlz_idx])
cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy').T
m_i, m_q, m_u = hp.synfast(cls, nside, fwhm=np.deg2rad(beam)/60, new=True)

cl_i_full = hp.anafast(m_i, lmax=lmax)
cl_i_bin = hp.anafast(m_i * bin_mask, lmax=lmax)
cl_i_apo = hp.anafast(m_i * apo_mask, lmax=lmax)

path_data = Path(f'./data/cmb')
path_data.mkdir(exist_ok=True, parents=True)
np.save(path_data / Path(f'I_full_{rlz_idx}.npy'), cl_i_full)
np.save(path_data / Path(f'I_bin_{rlz_idx}.npy'), cl_i_bin)
np.save(path_data / Path(f'I_apo_{rlz_idx}.npy'), cl_i_apo)

