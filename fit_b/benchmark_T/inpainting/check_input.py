import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

df = pd.read_csv('../mask/30.csv')
beam = 67
nside = 2048
npix = hp.nside2npix(nside)

noise_seed = np.load('../../seeds_noise_2k.npy')
cmb_seed = np.load('../../seeds_cmb_2k.npy')
fg_seed = np.load('../../seeds_fg_2k.npy')

rlz_idx = 0
flux_idx = 4

lon = np.rad2deg(df.at[flux_idx, 'lon'])
lat = np.rad2deg(df.at[flux_idx, 'lat'])

def check_map():
    m = hp.read_map(f'./input_pcfn/{rlz_idx}.fits')
    hp.gnomview(m, rot=[lon, lat, 0], xsize=400)
    plt.show()

def gen_fg_cl():
    Cl_TT = np.load('../../Cl_fg/data/cl_fg_TT.npy')
    Cl_EE = np.load('../../Cl_fg/data/cl_fg_EE.npy')
    Cl_BB = np.load('../../Cl_fg/data/cl_fg_BB.npy')
    Cl_TE = np.zeros_like(Cl_TT)
    return np.array([Cl_TT, Cl_EE, Cl_BB, Cl_TE])

def gen_map():
    ps = np.load('../data/ps/ps.npy')

    nstd = np.load('../../../FGSim/NSTDNORTH/2048/30.npy')
    np.random.seed(seed=noise_seed[rlz_idx])
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    print(f"{np.std(noise[1])=}")

    # cmb_iqu = np.load(f'../../fitdata/2048/CMB/215/{rlz_idx}.npy')
    # cls = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')
    cls = np.load('../../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    np.random.seed(seed=cmb_seed[rlz_idx])
    # cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=1999)
    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=3*nside-1)

    # l = np.arange(lmax+1)
    # cls_out = hp.anafast(cmb_iqu, lmax=lmax)

    cls_fg = gen_fg_cl()
    np.random.seed(seed=fg_seed[rlz_idx])
    fg_iqu = hp.synfast(cls_fg, nside=nside, fwhm=0, new=True, lmax=600)


    pcfn = noise + ps + cmb_iqu + fg_iqu
    # m = noise
    # cn = noise + cmb_iqu

    # m = np.load('./1_8k.npy')
    # np.save('./1_6k_pcn.npy', m)
    # np.save('./1_6k_cn.npy', cn)
    return pcfn, cmb_iqu, ps, fg_iqu, noise

def gen_cl():
    pcfn, c, p, f, n = gen_map()
    
    cls_pcfn = hp.anafast(pcfn)
    cls_f = hp.anafast(f)
    path_test = Path('./test')
    path_test.mkdir(exist_ok=True, parents=True)
    
    np.save(path_test / Path('pcfn_cls.npy'), cls_pcfn)
    np.save(path_test / Path('f_cls.npy'), cls_f)

def check_cl():
    cls_pcfn = np.load('./test/f_cls.npy')
    print(f'{cls_pcfn.shape=}')
    l = np.arange(np.size(cls_pcfn, axis=1))
    print(f'{l=}')
    plt.loglog(l, l*(l+1)*cls_pcfn[2] / (2*np.pi))
    plt.show()

check_cl()
