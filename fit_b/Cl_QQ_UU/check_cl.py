import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from pathlib import Path

lmax = 1000
l = np.arange(lmax + 1)

bl = hp.gauss_beam(fwhm=np.deg2rad(67)/60, lmax=lmax)

bin_mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
apo_mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')

cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy').T
bin_fsky = np.sum(bin_mask) / np.size(bin_mask)
apo_fsky = np.sum(apo_mask) / np.size(apo_mask)

cl_i_full_list = []
cl_i_bin_list = []
cl_i_apo_list = []
for rlz_idx in range(200):
    print(f'{rlz_idx=}')
    cl_i_full = np.load(f'./data/cmb/I_full_{rlz_idx}.npy')
    print(f'{cl_i_full[100]=}')
    cl_i_bin = np.load(f'./data/cmb/I_bin_{rlz_idx}.npy')
    cl_i_apo = np.load(f'./data/cmb/I_apo_{rlz_idx}.npy')

    cl_i_full_list.append(cl_i_full)
    cl_i_bin_list.append(cl_i_bin)
    cl_i_apo_list.append(cl_i_apo)

cl_i_full_arr = np.array(cl_i_full_list)
cl_i_bin_arr = np.array(cl_i_bin_list) / bin_fsky
cl_i_apo_arr = np.array(cl_i_apo_list) / apo_fsky

cl_i_full_mean = np.mean(cl_i_full_arr, axis=0)
cl_i_bin_mean = np.mean(cl_i_bin_arr, axis=0)
cl_i_apo_mean = np.mean(cl_i_apo_arr, axis=0)

plt.loglog(l*(l+1)*cl_i_full_mean, label='full')
plt.loglog(l*(l+1)*cl_i_bin_mean, label='bin')
plt.loglog(l*(l+1)*cl_i_apo_mean, label='apo')
plt.loglog(l*(l+1)*cls[0,:lmax+1] * bl**2, label='True')
plt.legend()
plt.show()



