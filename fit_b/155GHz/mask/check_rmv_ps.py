import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

from pathlib import Path
config_dir = Path(__file__).parent.parent
print(f'{config_dir=}')
sys.path.insert(0, str(config_dir))
from config import freq, lmax, nside, beam

noise_seed = np.load('../../seeds_noise_2k.npy')
cmb_seed = np.load('../../seeds_cmb_2k.npy')
fg_seed = np.load('../../seeds_fg_2k.npy')

npix = hp.nside2npix(nside=nside)
rlz_idx = 0

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
    n = noise
    return pcfn, noise

# m_pcfn, _ = gen_map(rlz_idx=rlz_idx)
# np.save('./pcfn.npy', m_pcfn)
m_pcfn = np.load('./pcfn.npy')
print(f'm_pcfn calc is ok!')

m = np.load(f'../fit_res/mean/3sigma/map_u_{rlz_idx}.npy')

df = pd.read_csv(f'../mask/{freq}_fit.csv')

flux_idx = 0
lon = np.rad2deg(df.at[flux_idx, 'lon'])
lat = np.rad2deg(df.at[flux_idx, 'lat'])

hp.gnomview(m_pcfn[2], rot=[lon, lat, 0], title='pcfn')
hp.gnomview(m, rot=[lon, lat, 0], title='rmv')
hp.gnomview(m - m_pcfn[2], rot=[lon, lat, 0], title='res')
plt.show()
