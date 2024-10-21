import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import pymaster as nmt
import glob

from pathlib import Path

lmax = 500
l = np.arange(lmax+1)
nside = 2048
rlz_idx = 0
threshold = 3

df = pd.read_csv('../../../FGSim/FreqBand')
freq = df.at[0, 'freq']
beam = df.at[0, 'beam']
print(f'{freq=}, {beam=}')

# rmv_list = []
# rmv1_list = []
# c_list = []
cf_list = []
cf_full_list = []
cfn_list = []
pcfn_list = []
rmv_qu_list = []

# ps_mask_list = []
qu_mask_list = []
# inp_eb_list = []

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

l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax, fold=0.2)
bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
ell_arr = bin_dl.get_effective_ells()

for rlz_idx in range(0,200):
    # if rlz_idx == 50:
        # continue

    n_qu = np.load(f'./pcfn_dl/QU/n/{rlz_idx}.npy')
    n_rc = np.load(f'./pcfn_dl/QU_cmb_slope/noise/{rlz_idx}.npy')

    pcfn = np.load(f'./pcfn_dl/QU/pcfn/{rlz_idx}.npy') - n_qu
    cfn = np.load(f'./pcfn_dl/QU/cfn/{rlz_idx}.npy') - n_qu
    cf = np.load(f'./pcfn_dl/QU/cf/{rlz_idx}.npy')
    cf_full = np.load(f'./pcfn_dl/RMV/cf/{rlz_idx}.npy')
    rmv_qu = np.load(f'./pcfn_dl/QU_cmb_slope/cmb/{rlz_idx}.npy') - n_rc
    qu_mask = np.load(f'./pcfn_dl/QU_cmb_slope/cfn/{rlz_idx}.npy') - n_rc

    # plt.plot(ell_arr, cf, label=f'cf {rlz_idx}')
    # plt.plot(ell_arr, cf1, label=f'cf1 full{rlz_idx}')
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
    cf_full_list.append(cf_full)
    cfn_list.append(cfn)
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
    # inp_eb_list.append(inp_eb)

# plt.show()

nsim = 200
# rmv_arr = np.array(rmv_list)
# c_arr = np.array(c_list)
cf_arr = np.array(cf_list)
cf_full_arr = np.array(cf_full_list)
cfn_arr = np.array(cfn_list)
pcfn_arr = np.array(pcfn_list)

rmv_qu_arr = np.array(rmv_qu_list)
# rmv_lon_lat_arr = np.array(rmv_lon_lat_list)
# rmv_b_qu_arr = np.array(rmv_b_qu_list)

# c_qu_arr = np.array(c_qu_list)
# cn_qu_arr = np.array(cn_qu_list)
# pcn_qu_arr = np.array(pcn_qu_list)

# ps_mask_arr = np.array(ps_mask_list)
qu_mask_arr = np.array(qu_mask_list)
# inp_eb_arr = np.array(inp_eb_list)


# pcfn_rmse_c = np.sqrt(np.sum((pcfn_arr-c_arr) ** 2, axis=0) / nsim)
# cfn_rmse_c = np.sqrt(np.sum((cfn_arr-c_arr) ** 2, axis=0) / nsim)
# cf_rmse_c = np.sqrt(np.sum((cf_arr-c_arr) ** 2, axis=0) / nsim)
# rmv_qu_rmse_c = np.sqrt(np.sum((rmv_qu_arr-c_arr) ** 2, axis=0) / nsim)
# inp_eb_rmse_c = np.sqrt(np.sum((inp_eb_arr-c_arr) ** 2, axis=0) / nsim)
# qu_mask_rmse_c = np.sqrt(np.sum((qu_mask_arr-c_arr) ** 2, axis=0) / nsim)

pcfn_mean = np.mean(pcfn_arr, axis=0)
cfn_mean = np.mean(cfn_arr, axis=0)
rmv_qu_mean = np.mean(rmv_qu_arr, axis=0)
cf_mean = np.mean(cf_arr, axis=0)
cf_full_mean = np.mean(cf_full_arr, axis=0)
qu_mask_mean = np.mean(qu_mask_arr, axis=0)

plt.figure(1)
plt.scatter(ell_arr, cf_mean, label='CMB + FG full sky SHT', marker='.')
plt.scatter(ell_arr, cf_full_mean, label='CMB + FG apo qu mask', marker='.')
plt.scatter(ell_arr, pcfn_mean, label='CMB + FG + PS + NOISE full sky SHT', marker='.')
plt.scatter(ell_arr, cfn_mean, label='CMB + FG + NOISE full sky SHT', marker='.')
plt.scatter(ell_arr, rmv_qu_mean, label='pcfn recycling', marker='.')
plt.scatter(ell_arr, qu_mask_mean, label='cfn recycling', marker='.')

plt.xlabel('$\\ell$')
plt.ylabel('$D_\\ell^{BB}$')
# plt.ylim(bottom=1e-10)
plt.loglog()
# plt.ylim(-0.1,0.1)
plt.legend()
plt.title('mean')

plt.figure(2)
plt.scatter(ell_arr, np.abs(cf_mean-cf_full_mean)/cf_mean, label='relative error', marker='.')
plt.xlabel('$\\ell$')
plt.ylabel('(qu mask - full SHT) / full SHT')
plt.loglog()
plt.legend()

# path_save = Path('/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20240918')
# path_save.mkdir(exist_ok=True, parents=True)
# plt.savefig(path_save / Path('pcfn_30GHz_76ps_rmse.png'), dpi=300)

plt.show()



