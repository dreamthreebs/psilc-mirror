import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import pymaster as nmt
import os,sys

from pathlib import Path
config_dir = Path(__file__).parent.parent
print(f'{config_dir=}')
sys.path.insert(0, str(config_dir))
from config import freq, lmax, nside, beam
l = np.arange(lmax+1)

rlz_idx=0
threshold = 3

df = pd.read_csv('../../../FGSim/FreqBand')
print(f'{freq=}, {beam=}')

bin_mask = np.load('../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
apo_mask = np.load('../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')
# ps_mask = np.load(f'../inpainting/mask/apo_ps_mask.npy')

noise_seeds = np.load('../../seeds_noise_2k.npy')
cmb_seeds = np.load('../../seeds_cmb_2k.npy')
fg_seeds = np.load('../../seeds_fg_2k.npy')

def gen_map(rlz_idx=0, mode='mean', return_noise=False):
    # mode can be mean or std
    noise_seed = np.load('../../seeds_noise_2k.npy')
    cmb_seed = np.load('../../seeds_cmb_2k.npy')
    nside = 2048

    nstd = np.load(f'../../../FGSim/NSTDNORTH/2048/{freq}.npy')
    npix = hp.nside2npix(nside=2048)
    np.random.seed(seed=noise_seed[rlz_idx])
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    print(f"{np.std(noise[1])=}")

    if return_noise:
        return noise

    ps = np.load(f'../../../fitdata/2048/PS/{freq}/ps.npy')
    fg = np.load(f'../../../fitdata/2048/FG/{freq}/fg.npy')

    cls = np.load('../../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    if mode=='std':
        np.random.seed(seed=cmb_seed[rlz_idx])
    elif mode=='mean':
        np.random.seed(seed=cmb_seed[0])

    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=3*nside-1)

    pcfn = noise + ps + cmb_iqu + fg
    cfn = noise + cmb_iqu + fg
    cf = cmb_iqu + fg
    n = noise
    return pcfn, cfn, cf, n


m_pcfn, m_cfn, m_cf, m_n= gen_map(rlz_idx=rlz_idx, mode='std')

pcfn_res = hp.fit_dipole(m=m_pcfn[1])
print(f'{pcfn_res=}')
cfn_res = hp.fit_dipole(m=m_cfn[1])
print(f'{cfn_res=}')
cf_res = hp.fit_dipole(m=m_cf[1])
print(f'{cf_res=}')
n_res = hp.fit_dipole(m=m_n[1])
print(f'{n_res=}')

