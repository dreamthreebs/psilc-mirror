import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os,sys
import pandas as pd

from pathlib import Path
from eblc_base_slope import EBLeakageCorrection

config_dir = Path(__file__).parent.parent
print(f'{config_dir=}')
sys.path.insert(0, str(config_dir))
from config import freq, lmax, nside, beam

npix = hp.nside2npix(nside)
df = pd.read_csv(f'../mask/{freq}.csv')
rlz_idx=0

noise_seed = np.load('../../seeds_noise_2k.npy')
cmb_seed = np.load('../../seeds_cmb_2k.npy')
fg_seed = np.load('../../seeds_fg_2k.npy')

def gen_fg_cl():
    cls_fg = np.load('../data/debeam_full_b/cl_fg.npy')
    Cl_TT = cls_fg[0]
    Cl_EE = cls_fg[1]
    Cl_BB = cls_fg[2]
    Cl_TE = np.zeros_like(Cl_TT)
    return np.array([Cl_TT, Cl_EE, Cl_BB, Cl_TE])

def gen_map(lmax, freq, beam):

    ps = np.load('../data/ps/ps.npy')

    nstd = np.load(f'../../../FGSim/NSTDNORTH/2048/{freq}.npy')
    np.random.seed(seed=noise_seed[rlz_idx])
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    print(f"{np.std(noise[1])=}")

    # cmb_iqu = np.load(f'../../fitdata/2048/CMB/215/{rlz_idx}.npy')
    # cls = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')
    cls = np.load('../../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    np.random.seed(seed=cmb_seed[rlz_idx])
    # cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=1999)
    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=lmax)

    # l = np.arange(lmax+1)
    # cls_out = hp.anafast(cmb_iqu, lmax=lmax)

    cls_fg = gen_fg_cl()
    np.random.seed(seed=fg_seed[rlz_idx])
    fg_iqu = hp.synfast(cls_fg, nside=nside, fwhm=0, new=True, lmax=lmax)


    m = noise + ps + cmb_iqu + fg_iqu
    cfn = cmb_iqu + fg_iqu + noise
    # m = noise
    # cn = noise + cmb_iqu

    # m = np.load('./1_8k.npy')
    # np.save('./1_6k_pcn.npy', m)
    # np.save('./1_6k_cn.npy', cn)
    return m, cfn, noise

def main():
    pcfn, cfn, n = gen_map(lmax, freq, beam)

    pcfn_b = hp.alm2map(hp.map2alm(pcfn)[2], nside=nside)
    cfn_b = hp.alm2map(hp.map2alm(cfn)[2], nside=nside)
    n_b = hp.alm2map(hp.map2alm(n)[2], nside=nside)
    np.save('./data/pcfn_b.npy', pcfn_b)
    np.save('./data/cfn_b.npy', cfn_b)
    np.save('./data/n_b.npy', n_b)

    mask = hp.read_map(f'./mask/mask.fits')

    # crt_pcfn_b_direct = hp.alm2map(hp.map2alm(pcfn*mask)[2], nside=nside)
    # np.save('./data/crt_pcfn_b_direct.npy', crt_pcfn_b_direct)

    # obj_eblc = EBLeakageCorrection(m=pcfn, lmax=3*nside-1, nside=nside, mask=mask, post_mask=mask, method='zzr')
    # crt_pcfn_b, _, cln_pcfn_b = obj_eblc.run_eblc()
    # np.save(f'./data/crt_pcfn_b.npy', crt_pcfn_b)
    # np.save(f'./data/cln_pcfn_b.npy', cln_pcfn_b)

def check_map():
    # crt_pcfn_b_direct = np.load('./data/crt_pcfn_b_direct.npy')
    crt_pcfn_b = np.load('./data/crt_pcfn_b.npy')
    cln_pcfn_b = np.load('./data/cln_pcfn_b.npy')
    pcfn_b = np.load('./data/pcfn_b.npy')
    cfn_b = np.load('./data/cfn_b.npy')

    mask = hp.read_map(f'./mask/mask.fits')
    mask1 = hp.read_map(f'./mask/mask_1dot8.fits')

    flux_idx = 4
    vmin = -2
    vmax = 2
    lon = np.rad2deg(df.at[flux_idx, 'lon'])
    lat = np.rad2deg(df.at[flux_idx, 'lat'])


    hp.gnomview(pcfn_b * mask, rot=[lon, lat, 0], title='pcfn * mask', min=vmin, max=vmax)
    hp.gnomview(crt_pcfn_b * mask, rot=[lon, lat, 0], title='crt_pcfn_b * mask', min=vmin, max=vmax)
    hp.gnomview(cln_pcfn_b * mask, rot=[lon, lat, 0], title='cln_pcfn_b * mask', min=vmin, max=vmax)

    # hp.gnomview(crt_pcfn_b_direct, rot=[lon, lat, 0], title='crt_pcfn_b_direct')
    # hp.gnomview(crt_pcfn_b, rot=[lon, lat, 0], title='crt_pcfn_b')
    # hp.gnomview(cln_pcfn_b, rot=[lon, lat, 0], title='cln_pcfn_b')
    # hp.gnomview(crt_pcfn_b - cln_pcfn_b, rot=[lon, lat, 0], title='crt - cln')
    # hp.gnomview(crt_pcfn_b - cfn_b, rot=[lon, lat, 0], title='crt - cfn')
    # hp.gnomview(cln_pcfn_b - cfn_b, rot=[lon, lat, 0], title='cln - cfn')
    hp.gnomview((pcfn_b - cfn_b)*mask1, rot=[lon, lat, 0], title='pcfn_b - cfn_b mask 1.8', min=vmin, max=vmax)
    hp.gnomview((pcfn_b - cfn_b)*mask, rot=[lon, lat, 0], title='pcfn_b - cfn_b mask 1.5', min=vmin, max=vmax)
    hp.gnomview(pcfn_b, rot=[lon, lat, 0], title='pcfn_b', min=vmin, max=vmax)

    plt.show()



# main()
check_map()
