import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import pymaster as nmt
import glob
import os,sys

from pathlib import Path
config_dir = Path(__file__).parent.parent
print(f'{config_dir=}')
sys.path.insert(0, str(config_dir))
from config import freq, lmax, nside, beam

l = np.arange(lmax+1)

df = pd.read_csv('../../../FGSim/FreqBand')
print(f'{freq=}, {beam=}')

cf_list = []
cfn_list = []
pcfn_list = []

rmv_list = []
ps_mask_list = []
inp_list = []

def generate_bins(l_min_start=30, delta_l_min=30, l_max=1500, fold=0.3):
    bins_edges = []
    l_min = l_min_start  # starting l_min

    while l_min < l_max:
        delta_l = max(delta_l_min, int(fold * l_min))
        l_next = l_min + delta_l
        bins_edges.append(l_min)
        l_min = l_next

    # Adding l_max to ensure the last bin goes up to l_max
    bins_edges.append(l_max)
    return bins_edges[:-1], bins_edges[1:]

l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax+1, fold=0.2)
bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
ell_arr = bin_dl.get_effective_ells()
print(f'{ell_arr=}')

for rlz_idx in range(1,200):
    n_qu = np.load(f'./pcfn_dl/MEAN/n/{rlz_idx}.npy')
    pcfn = np.load(f'./pcfn_dl/MEAN/pcfn/{rlz_idx}.npy') - n_qu
    cfn = np.load(f'./pcfn_dl/MEAN/cfn/{rlz_idx}.npy') - n_qu
    cf = np.load(f'./pcfn_dl/MEAN/cf/{rlz_idx}.npy')

    n_rmv = np.load(f'./pcfn_dl/RMV/n/{rlz_idx}.npy')
    rmv_qu = np.load(f'./pcfn_dl/RMV/MEAN/{rlz_idx}.npy') - n_rmv

    n_ps_mask = np.load(f'./pcfn_dl/PS_MASK/MEAN/n/{rlz_idx}.npy')
    ps_mask = np.load(f'./pcfn_dl/PS_MASK/MEAN/pcfn/{rlz_idx}.npy') - n_rmv


    n_inp = np.load(f'./pcfn_dl/INP/noise/{rlz_idx}.npy')
    inp = np.load(f'./pcfn_dl/INP/MEAN/{rlz_idx}.npy') - n_inp


    # plt.loglog(pcfn, label='pcfn')
    # plt.loglog(cfn, label='cfn')
    # plt.loglog(cf, label='cf')
    # plt.loglog(rmv_qu, label='rmv_qu')
    # plt.loglog(n_qu, label='n_qu')
    # plt.loglog(n_rmv, label='n_rmv')
    # plt.loglog(n_ps_mask, label='n_ps_mask')
    # plt.legend()
    # plt.show()

    cf_list.append(cf)
    cfn_list.append(cfn)
    pcfn_list.append(pcfn)

    rmv_list.append(rmv_qu)
    ps_mask_list.append(ps_mask)
    inp_list.append(inp)


pcfn_arr = np.asarray(pcfn_list)
cfn_arr = np.asarray(cfn_list)
cf_arr = np.asarray(cf_list)

rmv_arr = np.asarray(rmv_list)
ps_mask_arr = np.asarray(ps_mask_list)
inp_arr = np.asarray(inp_list)
print(f"{rmv_arr.shape=}")

nsim = np.size(pcfn_arr, axis=0)

pcfn_rmse = np.sqrt(np.sum((pcfn_arr-cf_arr) ** 2, axis=0) / nsim)
print(f'{pcfn_rmse.shape=}')
# cfn_rmse = np.sqrt(np.sum((cfn_arr-cf_arr) ** 2, axis=0) / nsim)
rmv_rmse = np.sqrt(np.sum((rmv_arr-cf_arr) ** 2, axis=0) / nsim)
# inp_eb_rmse = np.sqrt(np.sum((inp_eb_arr-cf_arr) ** 2, axis=0) / nsim)
ps_mask_rmse = np.sqrt(np.sum((ps_mask_arr-cf_arr) ** 2, axis=0) / nsim)
inp_rmse = np.sqrt(np.sum((inp_arr-cf_arr) ** 2, axis=0) / nsim)


plt.figure(1)
plt.scatter(ell_arr, pcfn_rmse, label='pcfn', marker='.')
# plt.scatter(ell_arr, cfn_rmse, label='cfn', marker='.')
plt.scatter(ell_arr, rmv_rmse, label='rmv', marker='.')
plt.scatter(ell_arr, ps_mask_rmse, label='ps_mask', marker='.')
plt.scatter(ell_arr, inp_rmse, label='inp', marker='.')
plt.xlabel('$\\ell$')
plt.ylabel('$D_\\ell^{BB} [\mu K^2]$')

plt.loglog()
plt.legend()
plt.title('RMSE(MEAN)')

plt.show()



