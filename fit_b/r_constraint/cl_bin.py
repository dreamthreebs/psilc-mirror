import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt

from pathlib import Path

"""
Try to see if the following binned cl is the same (only test on partial sky):
    1. pymaster scalar mode for B mode
    2. pymaster bin theoretical Cell

Result: still ~ 0.01 relative error, so directly use bin_cell is not recommanded
"""

# rlz_idx = 0

cls = np.load(f'../../src/cmbsim/cmbdata/cmbcl_8k.npy').T[0]
nside = 512
lmax = 2 * nside

# mask = np.load(f'../../src/mask/north/APOMASKC1_5.npy')
mask = np.ones(hp.nside2npix(nside))
# hp.orthview(mask, rot=[100,50,0])
# plt.show()

bin_dl = nmt.bins.NmtBin.from_lmax_linear(lmax=lmax, nlb=30, is_Dell=True)
ell_arr = bin_dl.get_effective_ells()

def gen_sim(rlz_idx):
    np.random.seed(rlz_idx)
    cmb = hp.synfast(cls, nside=nside)

    # hp.mollview(cmb)
    # plt.show()
    return cmb

def calc_dl_from_scalar_map(scalar_map, apo_mask, bin_dl, masked_on_input):
    scalar_field = nmt.NmtField(apo_mask, [scalar_map], masked_on_input=masked_on_input, lmax=lmax, lmax_mask=lmax)
    dl = nmt.compute_full_master(scalar_field, scalar_field, bin_dl)
    return dl[0]

def try_sca_mode(rlz_idx):
    print(f'try scalar mode')

    cmb = gen_sim(rlz_idx)
    dl = calc_dl_from_scalar_map(scalar_map=cmb, apo_mask=mask, bin_dl=bin_dl, masked_on_input=False)
    Path('./test_data/cmb_dl_full').mkdir(exist_ok=True, parents=True)
    np.save(f'./test_data/cmb_dl_full/{rlz_idx}.npy', dl)

def try_bin_cell():
    print(f'try bin cell')

    dl = bin_dl.bin_cell(cls_in=cls[:lmax+1])
    # plt.loglog(ell_arr, dl)
    # plt.show()
    return dl

def calc_dl():
    for i in range(0,10000):
        print(f'{i=}')
        try_sca_mode(rlz_idx=i)


def cpr_res():

    dl_bin_cell = try_bin_cell()
    dl_sca = np.mean([np.load(f'./test_data/cmb_dl/{rlz_idx}.npy') for rlz_idx in range(200)], axis=0)
    dl_sca_full = np.mean([np.load(f'./test_data/cmb_dl_full/{rlz_idx}.npy') for rlz_idx in range(200)], axis=0)

    plt.loglog(ell_arr, dl_bin_cell, label='bin cell')
    plt.loglog(ell_arr, dl_sca, label='scalar mode')
    plt.loglog(ell_arr, dl_sca_full, label='scalar mode with full sky')
    plt.loglog(ell_arr, np.abs(dl_sca-dl_bin_cell)/dl_bin_cell, label='relative error')
    plt.loglog(ell_arr, np.abs(dl_sca_full-dl_bin_cell)/dl_bin_cell, label='relative error for full sky')
    plt.legend()
    plt.show()

# try_bin_cell()
# calc_dl()
cpr_res()
