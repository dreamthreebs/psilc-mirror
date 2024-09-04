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
c_list = []
cn_list = []
pcn_list = []

rmv_qu_list = []
rmv_lon_lat_list = []
# rmv_b_qu_list = []

# rmv1_list = []
c_qu_list = []
cn_qu_list = []
pcn_qu_list = []

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

l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax, fold=0.2)
bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
ell_arr = bin_dl.get_effective_ells()

nsim = 201

for rlz_idx in range(0,200):
    # if rlz_idx == 50:
        # continue
    # n = np.load(f'./pcn_dl/B/n/{rlz_idx}.npy')
    n_qu = np.load(f'./pcn_dl/QU/n/{rlz_idx}.npy')
    n_rmv = np.load(f'./pcn_dl/QU_edge/removal_3sigma/{rlz_idx}.npy')
    # n_lon_lat = np.load(f'./pcn_dl/QU_edge/removal_3sigma_n/{rlz_idx}.npy')
    n_inp = np.load(f'./pcn_dl/INP_SMALL_B/inpaint_eb_3sigma/{rlz_idx}.npy')
    # n_ps_mask = np.load(f'./pcn_dl/PS_MASK_n/ps_3sigma/{rlz_idx}.npy')
    # n_ps_mask = np.load(f'./pcn_dl/INP_SMALL_B/inpaint_eb_3sigma_n/{rlz_idx}.npy')
    n_qu_mask = np.load(f'./pcn_dl/ACT/curl_yp_ps_n/{rlz_idx}.npy')

    # rmv = np.load(f'./pcn_dl/B/removal_3sigma/{rlz_idx}.npy') - n
    rmv_qu = np.load(f'./pcfn_dl/QU_edge/removal_3sigma/{rlz_idx}.npy') - n_rmv
    # rmv_lon_lat = np.load(f'./pcn_dl/QU_edge/removal_3sigma/{rlz_idx}.npy') - n_lon_lat
    # rmv_b_qu = np.load(f'./pcn_dl/QU_B/removal_3sigma/{rlz_idx}.npy') - n_qu
    # rmv1 = np.load(f'./pcn_dl/B/removal_10sigma/{rlz_idx}.npy')
    c = np.load(f'./pcfn_dl/QU_edge/QU/c/{rlz_idx}.npy')
    # c_qu = np.load(f'./pcn_dl/QU/c/{rlz_idx}.npy')
    cn = np.load(f'./pcn_dl/QU/cn/{rlz_idx}.npy') - n_qu
    # cn_qu = np.load(f'./pcn_dl/QU/cn/{rlz_idx}.npy') - n_qu
    pcn = np.load(f'./pcn_dl/QU/pcn/{rlz_idx}.npy') - n_qu
    # pcn_qu = np.load(f'./pcn_dl/QU/pcn/{rlz_idx}.npy') - n_qu
    # ps_mask = np.load(f'./pcn_dl/PS_MASK/ps_3sigma/{rlz_idx}.npy') - n_ps_mask
    ps_mask = np.load(f'./pcn_dl/INP_SMALL_B/inpaint_eb_3sigma/{rlz_idx}.npy') - n_inp
    qu_mask = np.load(f'./pcn_dl/ACT/curl_yp_ps/{rlz_idx}.npy') - n_qu_mask
    # inp_qu = np.load(f'./pcn_dl/INP_QU_1/inpaint_qu_3sigma/{rlz_idx}.npy') - n_qu
    inp_eb = np.load(f'./pcn_dl/INP_B_1/inpaint_eb_3sigma/{rlz_idx}.npy') - n_inp

    # plt.plot(ell_arr, c, label=f'c {rlz_idx}')
    # plt.plot(ell_arr, n_rmv, label=f'rmv n{rlz_idx}')
    # plt.plot(ell_arr, n_qu, label=f'n {rlz_idx}')
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
    c_list.append(c)
    cn_list.append(cn)
    pcn_list.append(pcn)

    rmv_qu_list.append(rmv_qu)
    rmv_lon_lat_list.append(rmv_lon_lat)
    # rmv_b_qu_list.append(rmv_b_qu)
    # c_qu_list.append(c_qu)
    # cn_qu_list.append(cn_qu)
    # pcn_qu_list.append(pcn_qu)

    ps_mask_list.append(ps_mask)
    qu_mask_list.append(qu_mask)
    # inp_qu_list.append(inp_qu)
    inp_eb_list.append(inp_eb)

# plt.show()

# rmv_arr = np.array(rmv_list)
c_arr = np.array(c_list)
cn_arr = np.array(cn_list)
pcn_arr = np.array(pcn_list)
rmv_qu_arr = np.array(rmv_qu_list)
rmv_lon_lat_arr = np.array(rmv_lon_lat_list)
# rmv_b_qu_arr = np.array(rmv_b_qu_list)
# c_qu_arr = np.array(c_qu_list)
# cn_qu_arr = np.array(cn_qu_list)
# pcn_qu_arr = np.array(pcn_qu_list)
ps_mask_arr = np.array(ps_mask_list)
qu_mask_arr = np.array(qu_mask_list)
# inp_qu_arr = np.array(inp_qu_list)
inp_eb_arr = np.array(inp_eb_list)
# print(f'{rmv_arr.shape=}')

pcn_rmse = np.sqrt(np.sum((pcn_arr-c_arr) ** 2, axis=0) / nsim)
print(f'{pcn_rmse.shape=}')
cn_rmse = np.sqrt(np.sum((cn_arr-c_arr) ** 2, axis=0) / nsim)
rmv_qu_rmse = np.sqrt(np.sum((rmv_qu_arr-c_arr) ** 2, axis=0) / nsim)
rmv_lon_lat_rmse = np.sqrt(np.sum((rmv_lon_lat_arr-c_arr) ** 2, axis=0) / nsim)
inp_eb_rmse = np.sqrt(np.sum((inp_eb_arr-c_arr) ** 2, axis=0) / nsim)
ps_mask_rmse = np.sqrt(np.sum((ps_mask_arr-c_arr) ** 2, axis=0) / nsim)
qu_mask_rmse = np.sqrt(np.sum((qu_mask_arr-c_arr) ** 2, axis=0) / nsim)

c_mean = np.mean(c_arr, axis=0)
print(f'{ell_arr[1:7]=}')
pcn_rmse_ratio = np.sum(pcn_rmse[1:7] / c_mean[1:7])
cn_rmse_ratio = np.sum(cn_rmse[1:7] / c_mean[1:7])
rmv_qu_rmse_ratio = np.sum(rmv_qu_rmse[1:7] / c_mean[1:7])
rmv_lon_lat_rmse_ratio = np.sum(rmv_lon_lat_rmse[1:7] / c_mean[1:7])
inp_eb_rmse_ratio = np.sum(inp_eb_rmse[1:7] / c_mean[1:7])
ps_mask_rmse_ratio = np.sum(ps_mask_rmse[1:7] / c_mean[1:7])
qu_mask_rmse_ratio = np.sum(qu_mask_rmse[1:7] / c_mean[1:7])
print(f'{pcn_rmse_ratio=}')
print(f'{cn_rmse_ratio=}')
print(f'{rmv_qu_rmse_ratio=}')
print(f'{rmv_lon_lat_rmse_ratio=}')
print(f'{inp_eb_rmse_ratio=}')
print(f'{ps_mask_rmse_ratio=}')
print(f'{qu_mask_rmse_ratio=}')

plt.scatter(ell_arr, c_mean, label='input CMB power spectrum (not RMSE)', marker='.')
plt.scatter(ell_arr, pcn_rmse, label='No point source treatment', marker='.')
plt.scatter(ell_arr, cn_rmse, label='CMB + noise', marker='.')
plt.scatter(ell_arr, rmv_qu_rmse, label='Template fitting method with fixed lon lat', marker='.')
plt.scatter(ell_arr, rmv_lon_lat_rmse, label='Template fitting method with free lon lat', marker='.')
# plt.scatter(ell_arr, rmv_qu_rmse, label='rmv b qu', marker='.')
plt.scatter(ell_arr, inp_eb_rmse, label='Recycling method + inpaint on B', marker='.')
# plt.scatter(ell_arr, inp_qu_rmse, label='inp qu ', marker='.')
plt.scatter(ell_arr, ps_mask_rmse, label='ps mask ', marker='.')
plt.scatter(ell_arr, qu_mask_rmse, label='Apodized QU mask', marker='.')
plt.xlabel('$\\ell$')
plt.ylabel('$D_\\ell^{BB}$')
# plt.ylim(bottom=1e-10)
plt.loglog()
# plt.ylim(-0.1,0.1)
plt.legend()
plt.title('RMSE')


# plt.savefig('/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20240507/30GHz_1.png', dpi=300)

plt.show()




