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

# rmv_list = []
# rmv1_list = []
# c_list = []
cf_list = []
cfn_list = []
pcfn_list = []

rmv_qu_list = []
rmv_lon_lat_list = []
# rmv_b_qu_list = []

# # rmv1_list = []
# cf_qu_list = []
# cfn_qu_list = []
# pcfn_qu_list = []

ps_mask_list = []
qu_mask_list = []
# inp_qu_list = []
inp_eb_list = []

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

for rlz_idx in range(0,200):
    # if rlz_idx == 50:
        # continue
    # n = np.load(f'./pcn_dl/B/n/{rlz_idx}.npy')
    # n_qu = np.load(f'../../benchmark_T/fit_res/pcfn_dl/QU/n/{rlz_idx}.npy')
    n_pcfn = np.load(f'./pcfn_dl/RMV/n/{rlz_idx}.npy')
    n_rmv = np.load(f'./pcfn_dl/RMV/rmv_n/{rlz_idx}.npy')
    # n_lon_lat = np.load(f'./pcn_dl/QU_lon_lat_n/removal_3sigma/{rlz_idx}.npy')
    n_inp = np.load(f'./pcfn_dl/RMV/inp_m2_n/{rlz_idx}.npy')
    # n_ps_mask = np.load(f'./pcfn_dl/QU_fg_slope/removal_n_3sigma/{rlz_idx}.npy')
    # n_ps_mask = np.load(f'./pcfn_dl/RMV/n/{rlz_idx}.npy')
    n_qu_mask = np.load(f'./pcfn_dl/RMV/apo_n/{rlz_idx}.npy')
    # rmv = np.load(f'./pcn_dl/QU/removal_3sigma/{rlz_idx}.npy') - n_rmv
    rmv_qu = np.load(f'./pcfn_dl/RMV/rmv/{rlz_idx}.npy') - n_rmv
    # rmv_lon_lat = np.load(f'./pcn_dl/QU_lon_lat/removal_3sigma/{rlz_idx}.npy') - n_lon_lat
    # rmv_b_qu = np.load(f'./pcn_dl/QU_B/removal_3sigma/{rlz_idx}.npy') - n_qu
    # rmv1 = np.load(f'./pcn_dl/B/removal_10sigma/{rlz_idx}.npy')
    # c = np.load(f'./pcfn_dl/QU/c/{rlz_idx}.npy')
    # cf = np.load(f'../../benchmark_T/fit_res/pcfn_dl/QU/cf/{rlz_idx}.npy')
    cf = np.load(f'./pcfn_dl/RMV/cf/{rlz_idx}.npy')
    # c_qu = np.load(f'./pcn_dl/QU/c/{rlz_idx}.npy')
    # cfn = np.load(f'../../benchmark_T/fit_res/pcfn_dl/QU/cfn/{rlz_idx}.npy') - n_qu
    # cfn = np.load(f'./pcfn_dl/RMV/pcfn/{rlz_idx}.npy') - n_pcfn
    # cfn = np.load(f'../../benchmark_T/fit_res/pcfn_dl/QU/cfn/{rlz_idx}.npy') - n_qu
    # cfn = np.load(f'./pcfn_dl/QU/cfn/{rlz_idx}.npy') - n_pcfn
    # cn_qu = np.load(f'./pcn_dl/QU/cn/{rlz_idx}.npy') - n_qu
    # pcfn = np.load(f'./pcfn_dl/QU_cmb_slope/cmb/{rlz_idx}.npy') - n_pcfn
    pcfn = np.load(f'./pcfn_dl/RMV/pcfn/{rlz_idx}.npy') - n_pcfn
    # pcfn_1 = np.load(f'./pcfn_dl/RMV/pcfn_bin/{rlz_idx}.npy') - n_pcfn
    # pcfn_full = np.load(f'../../benchmark_T/fit_res/pcfn_dl/QU/pcfn/{rlz_idx}.npy') - n_qu
    # pcn_qu = np.load(f'./pcn_dl/QU/pcn/{rlz_idx}.npy') - n_qu
    # ps_mask = np.load(f'./pcfn_dl/QU/cfn/{rlz_idx}.npy') - n_ps_mask
    qu_mask = np.load(f'./pcfn_dl/RMV/apo/{rlz_idx}.npy') - n_qu_mask
    # inp_qu = np.load(f'./pcn_dl/INP_QU_1/inpaint_qu_3sigma/{rlz_idx}.npy') - n_qu
    inp_eb = np.load(f'./pcfn_dl/RMV/inp_m2/{rlz_idx}.npy') - n_inp

    # plt.plot(ell_arr, pcfn, label=f'pcfn {rlz_idx}')
    # # plt.plot(ell_arr, cfn, label=f'cfn {rlz_idx}')
    # plt.plot(ell_arr, cf, label=f'cf {rlz_idx}')

    # plt.plot(ell_arr, n_rmv, label=f'rmv n{rlz_idx}')
    # plt.plot(ell_arr, n_pcfn, label=f'pcfn n{rlz_idx}')
    # # plt.plot(ell_arr, n_qu, label=f'n {rlz_idx}')
    # plt.plot(ell_arr, rmv_qu, label=f'rmv_qu {rlz_idx}')
    # plt.plot(ell_arr, pcfn, label=f'pcfn {rlz_idx}')
    # # plt.plot(ell_arr, cfn, label=f'cfn {rlz_idx}')

    # plt.plot(ell_arr, pcfn_full, label=f'pcfn full{rlz_idx}')
    # plt.plot(ell_arr, inp_eb, label=f'inp_eb {rlz_idx}')
    # plt.plot(ell_arr, n_inp, label=f'inp n {rlz_idx}')
    # plt.plot(ell_arr, n_ps_mask, label=f'ps mask n {rlz_idx}')
    # plt.plot(ell_arr, n_qu_mask, label=f'qu mask n {rlz_idx}')
    # plt.plot(ell_arr, n_lon_lat, label=f'rmv lon lat n {rlz_idx}')
    # plt.plot(ell_arr, np.abs(n_rmv - n_qu), label=f'rmv - n {rlz_idx}')
    # plt.plot(ell_arr, np.abs(n_lon_lat - n_qu), label=f'rmv lon lat - n {rlz_idx}')
    # plt.plot(ell_arr, np.abs(n_inp - n_qu), label=f'inp - n {rlz_idx}')
    # plt.plot(ell_arr, np.abs(n_ps_mask - n_qu), label=f'ps mask - n {rlz_idx}')
    # plt.plot(ell_arr, np.abs(n_qu_mask - n_qu), label=f'qu mask - n {rlz_idx}')

    # plt.semilogy()
    # plt.legend()
    # plt.show()

    # rmv_list.append(rmv)
    # c_list.append(c)
    cf_list.append(cf)
    # cfn_list.append(cfn)
    pcfn_list.append(pcfn)

    rmv_qu_list.append(rmv_qu)
    # rmv_lon_lat_list.append(rmv_lon_lat)
    # rmv_b_qu_list.append(rmv_b_qu)
    # c_qu_list.append(c_qu)
    # cn_qu_list.append(cn_qu)
    # pcn_qu_list.append(pcn_qu)

    # ps_mask_list.append(ps_mask)
    qu_mask_list.append(qu_mask)
    # inp_qu_list.append(inp_qu)
    inp_eb_list.append(inp_eb)

# plt.show()

nsim = 200
# rmv_arr = np.array(rmv_list)
# c_arr = np.array(c_list)
cf_arr = np.array(cf_list)
# cfn_arr = np.array(cfn_list)
pcfn_arr = np.array(pcfn_list)

rmv_qu_arr = np.array(rmv_qu_list)
# rmv_lon_lat_arr = np.array(rmv_lon_lat_list)
# rmv_b_qu_arr = np.array(rmv_b_qu_list)

# c_qu_arr = np.array(c_qu_list)
# cn_qu_arr = np.array(cn_qu_list)
# pcn_qu_arr = np.array(pcn_qu_list)

# ps_mask_arr = np.array(ps_mask_list)
qu_mask_arr = np.array(qu_mask_list)
# inp_qu_arr = np.array(inp_qu_list)
inp_eb_arr = np.array(inp_eb_list)
# print(f'{rmv_arr.shape=}')


# pcfn_rmse_c = np.sqrt(np.sum((pcfn_arr-c_arr) ** 2, axis=0) / nsim)
# cfn_rmse_c = np.sqrt(np.sum((cfn_arr-c_arr) ** 2, axis=0) / nsim)
# cf_rmse_c = np.sqrt(np.sum((cf_arr-c_arr) ** 2, axis=0) / nsim)
# rmv_qu_rmse_c = np.sqrt(np.sum((rmv_qu_arr-c_arr) ** 2, axis=0) / nsim)
# inp_eb_rmse_c = np.sqrt(np.sum((inp_eb_arr-c_arr) ** 2, axis=0) / nsim)
# qu_mask_rmse_c = np.sqrt(np.sum((qu_mask_arr-c_arr) ** 2, axis=0) / nsim)

pcfn_rmse = np.sqrt(np.sum((pcfn_arr-cf_arr) ** 2, axis=0) / nsim)[:13]
print(f'{pcfn_rmse.shape=}')
# cfn_rmse = np.sqrt(np.sum((cfn_arr-cf_arr) ** 2, axis=0) / nsim)[:13]
rmv_qu_rmse = np.sqrt(np.sum((rmv_qu_arr-cf_arr) ** 2, axis=0) / nsim)[:13]
# rmv_lon_lat_rmse = np.sqrt(np.sum((rmv_lon_lat_arr-c_arr) ** 2, axis=0) / nsim)
inp_eb_rmse = np.sqrt(np.sum((inp_eb_arr-cf_arr) ** 2, axis=0) / nsim)[:13]
# ps_mask_rmse = np.sqrt(np.sum((ps_mask_arr-cf_arr) ** 2, axis=0) / nsim)
qu_mask_rmse = np.sqrt(np.sum((qu_mask_arr-cf_arr) ** 2, axis=0) / nsim)[:13]

cf_mean = np.mean(cf_arr, axis=0)[:13]
print(f'{ell_arr[1:7]=}')
pcfn_rmse_ratio = np.sum(pcfn_rmse[1:7] / cf_mean[1:7])
# cfn_rmse_ratio = np.sum(cfn_rmse[1:7] / cf_mean[1:7])
rmv_qu_rmse_ratio = np.sum(rmv_qu_rmse[1:7] / cf_mean[1:7])
# rmv_lon_lat_rmse_ratio = np.sum(rmv_lon_lat_rmse[1:7] / c_mean[1:7])
inp_eb_rmse_ratio = np.sum(inp_eb_rmse[1:7] / cf_mean[1:7])
# ps_mask_rmse_ratio = np.sum(ps_mask_rmse[1:7] / cf_mean[1:7])
qu_mask_rmse_ratio = np.sum(qu_mask_rmse[1:7] / cf_mean[1:7])
print(f'{pcfn_rmse_ratio=}')
# print(f'{cfn_rmse_ratio=}')
print(f'{rmv_qu_rmse_ratio=}')
# print(f'{rmv_lon_lat_rmse_ratio=}')
print(f'{inp_eb_rmse_ratio=}')
# print(f'{ps_mask_rmse_ratio=}')
print(f'{qu_mask_rmse_ratio=}')

ell_arr = ell_arr[:13]
plt.figure(1)
plt.scatter(ell_arr, cf_mean, label='input CMB + FG power spectrum (True value, not RMSE)', marker='.', color='black')
plt.scatter(ell_arr, pcfn_rmse, label='PS + FG + CMB + NOISE (Baseline)', marker='.')
# plt.scatter(ell_arr, cfn_rmse, label='FG + CMB + NOISE', marker='.')
plt.scatter(ell_arr, rmv_qu_rmse, label='Template fitting method with fixed lon lat', marker='.')
# plt.scatter(ell_arr, rmv_lon_lat_rmse, label='template fitting method with free lon lat', marker='.')
# plt.scatter(ell_arr, rmv_qu_rmse, label='rmv b qu', marker='.')
plt.scatter(ell_arr, inp_eb_rmse, label='Recycling method + inpaint on B', marker='.')
# plt.scatter(ell_arr, inp_qu_rmse, label='inp qu ', marker='.')
# plt.scatter(ell_arr, ps_mask_rmse, label='no EB leakage, Template Fitting', marker='.')
plt.scatter(ell_arr, qu_mask_rmse, label='Mask point sources holes on QU', marker='.')

plt.xlabel('$\\ell$')
plt.ylabel('$\\Delta D_\\ell^{BB} [\mu K^2]$')
# plt.ylim(bottom=1e-10)
plt.loglog()
# plt.ylim(-0.1,0.1)
plt.legend()
plt.title('RMSE')

path_save = Path('/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20241109')
path_save.mkdir(exist_ok=True, parents=True)
plt.savefig(path_save / Path('95GHz_62ps_rmse.png'), dpi=300)


plt.figure(2)
plt.scatter(ell_arr, pcfn_rmse/rmv_qu_rmse, marker='.', label='PS + FG + CMB + NOISE (Baseline) / Template fitting method')
plt.scatter(ell_arr, inp_eb_rmse/rmv_qu_rmse, marker='.', label='Recycling method + inpaint on B / Template fitting method')
plt.scatter(ell_arr, qu_mask_rmse/rmv_qu_rmse, marker='.', label='Mask point sources holes on QU / Template fitting method')
plt.loglog()
plt.legend()
plt.xlabel('$\\ell$')
plt.ylabel('$\\Delta D_\\ell^{BB}$ ratio')
plt.title('comparison between different methods')

path_save = Path('/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20241109')
path_save.mkdir(exist_ok=True, parents=True)
plt.savefig(path_save / Path('cpr_results_95GHz_62ps.png'), dpi=300)

print(f'{np.mean(pcfn_rmse/rmv_qu_rmse)=}')
print(f'{np.mean(inp_eb_rmse/rmv_qu_rmse)=}')
print(f'{np.mean(qu_mask_rmse/rmv_qu_rmse)=}')

plt.show()





