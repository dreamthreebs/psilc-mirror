import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from pathlib import Path

nside = 2048
freq_list = [30, 95, 155, 215, 270]
beam_list = [67, 30, 17, 11, 9]
beam_base = 17 #arcmin
lmax = 1500
nside = 2048
rlz_idx = 0

cmb_seed = np.load(f'../seeds_cmb_2k.npy')
noise_seed = np.load(f'../seeds_noise_2k.npy')

""" Check if it is the effect from smooth"""
def gen_map(freq, rlz_idx):
    # generate CMB + diffuse FG + noise at different freq
    nstd = np.load(f'../../../FGSim/NSTDNORTH/2048/{freq}.npy')
    npix = hp.nside2npix(nside=nside)
    np.random.seed(seed=noise_seed[rlz_idx])
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    print(f"{np.std(noise[1])=}")

    fg = np.load(f'../../fitdata/2048/FG/{freq}/fg.npy')
    cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    np.random.seed(seed=cmb_seed[0])
    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=3*nside-1)


    pass

def smooth_to_B():
    # deconvolve and convolve to 17 arcmin
    pass

def do_nilc():
    pass

