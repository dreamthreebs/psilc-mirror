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

for rlz_idx in range(1,200):
    # if rlz_idx == 50:
        # continue
    # n = np.load(f'./pcn_dl/B/n/{rlz_idx}.npy')
    n_qu = np.load(f'./pcn_dl/QU/n/{rlz_idx}.npy')
    # n_rmv = np.load(f'./pcn_dl/QU_n/removal_3sigma/{rlz_idx}.npy')
    # n_lon_lat = np.load(f'./pcn_dl/QU_lon_lat_n/removal_3sigma/{rlz_idx}.npy')
    n_inp = np.load(f'./pcn_dl/INP_B/inpaint_eb_3sigma_n/{rlz_idx}.npy')
    # n_ps_mask = np.load(f'./pcn_dl/PS_MASK_n/ps_3sigma/{rlz_idx}.npy')
    # n_qu_mask = np.load(f'./pcn_dl/ACT/curl_yp_ps_n/{rlz_idx}.npy')
    # rmv = np.load(f'./pcn_dl/B/removal_3sigma/{rlz_idx}.npy') - n
    # rmv_qu = np.load(f'./pcn_dl/QU/removal_3sigma/{rlz_idx}.npy') - n_rmv
    # rmv_lon_lat = np.load(f'./pcn_dl/QU_lon_lat/removal_3sigma/{rlz_idx}.npy') - n_lon_lat
    # rmv_b_qu = np.load(f'./pcn_dl/QU_B/removal_3sigma/{rlz_idx}.npy') - n_qu
    # rmv1 = np.load(f'./pcn_dl/B/removal_10sigma/{rlz_idx}.npy')
    c = np.load(f'./pcn_dl/QU/c/{rlz_idx}.npy')
    # c_qu = np.load(f'./pcn_dl/QU/c/{rlz_idx}.npy')
    cn = np.load(f'./pcn_dl/QU/cn/{rlz_idx}.npy') - n_qu
    # cn_qu = np.load(f'./pcn_dl/QU/cn/{rlz_idx}.npy') - n_qu
    pcn = np.load(f'./pcn_dl/QU/pcn/{rlz_idx}.npy') - n_qu
    # pcn_qu = np.load(f'./pcn_dl/QU/pcn/{rlz_idx}.npy') - n_qu
    # ps_mask = np.load(f'./pcn_dl/PS_MASK/ps_3sigma/{rlz_idx}.npy') - n_qu
    # qu_mask = np.load(f'./pcn_dl/ACT/curl_yp_ps/{rlz_idx}.npy') - n_qu
    # inp_qu = np.load(f'./pcn_dl/INP_QU_1/inpaint_qu_3sigma/{rlz_idx}.npy') - n_qu
    inp_eb = np.load(f'./pcn_dl/INP_B/inpaint_eb_3sigma/{rlz_idx}.npy') - n_inp

    # plt.plot(ell_arr, c, label=f'c {rlz_idx}')
    # # plt.plot(ell_arr, n_rmv, label=f'rmv n{rlz_idx}')
    # plt.plot(ell_arr, n_qu, label=f'n {rlz_idx}')
    # plt.plot(ell_arr, n_inp, label=f'inp n {rlz_idx}')
    # # plt.plot(ell_arr, n_ps_mask, label=f'ps mask n {rlz_idx}')
    # # plt.plot(ell_arr, n_qu_mask, label=f'qu mask n {rlz_idx}')
    # # plt.plot(ell_arr, n_lon_lat, label=f'rmv lon lat n {rlz_idx}')
    # # plt.plot(ell_arr, np.abs(n_rmv - n_qu), label=f'rmv - n {rlz_idx}')
    # # plt.plot(ell_arr, np.abs(n_lon_lat - n_qu), label=f'rmv lon lat - n {rlz_idx}')
    # plt.plot(ell_arr, np.abs(n_inp - n_qu), label=f'inp - n {rlz_idx}')
    # # plt.plot(ell_arr, np.abs(n_ps_mask - n_qu), label=f'ps mask - n {rlz_idx}')
    # # plt.plot(ell_arr, np.abs(n_qu_mask - n_qu), label=f'qu mask - n {rlz_idx}')
    # plt.semilogy()
    # plt.legend()
    # plt.show()

    # rmv_list.append(rmv)
    c_list.append(c)
    cn_list.append(cn)
    pcn_list.append(pcn)

    # rmv_qu_list.append(rmv_qu)
    # rmv_lon_lat_list.append(rmv_lon_lat)
    # rmv_b_qu_list.append(rmv_b_qu)
    # c_qu_list.append(c_qu)
    # cn_qu_list.append(cn_qu)
    # pcn_qu_list.append(pcn_qu)

    # ps_mask_list.append(ps_mask)
    # qu_mask_list.append(qu_mask)
    # inp_qu_list.append(inp_qu)
    inp_eb_list.append(inp_eb)

# plt.show()

# rmv_arr = np.array(rmv_list)
c_arr = np.array(c_list)
cn_arr = np.array(cn_list)
pcn_arr = np.array(pcn_list)

# rmv_qu_arr = np.array(rmv_qu_list)
# rmv_lon_lat_arr = np.array(rmv_lon_lat_list)
# rmv_b_qu_arr = np.array(rmv_b_qu_list)

# c_qu_arr = np.array(c_qu_list)
# cn_qu_arr = np.array(cn_qu_list)
# pcn_qu_arr = np.array(pcn_qu_list)

# ps_mask_arr = np.array(ps_mask_list)
# qu_mask_arr = np.array(qu_mask_list)
# inp_qu_arr = np.array(inp_qu_list)
inp_eb_arr = np.array(inp_eb_list)
# print(f'{rmv_arr.shape=}')

# rmv_mean = np.mean(rmv_arr, axis=0)
c_mean = np.mean(c_arr, axis=0)
cn_mean = np.mean(cn_arr, axis=0)
pcn_mean = np.mean(pcn_arr, axis=0)

# rmv_qu_mean = np.mean(rmv_qu_arr, axis=0)
# rmv_lon_lat_mean = np.mean(rmv_lon_lat_arr, axis=0)
# rmv_b_qu_mean = np.mean(rmv_b_qu_arr, axis=0)

# c_qu_mean = np.mean(c_qu_arr, axis=0)
# cn_qu_mean = np.mean(cn_qu_arr, axis=0)
# pcn_qu_mean = np.mean(pcn_qu_arr, axis=0)

# ps_mask_mean = np.mean(ps_mask_arr, axis=0)
# qu_mask_mean = np.mean(qu_mask_arr, axis=0)
# inp_qu_mean = np.mean(inp_qu_arr, axis=0)
inp_eb_mean = np.mean(inp_eb_arr, axis=0)
# print(f'{rmv_mean.shape=}')

# rmv_std = np.std(rmv_arr, axis=0)
# rmv1_std = np.std(rmv1_arr, axis=0)
c_std = np.std(c_arr, axis=0)
cn_std = np.std(cn_arr, axis=0)
pcn_std = np.std(pcn_arr, axis=0)

# rmv_qu_std = np.std(rmv_qu_arr, axis=0)
# rmv_lon_lat_std = np.std(rmv_lon_lat_arr, axis=0)
# rmv_b_qu_std = np.std(rmv_b_qu_arr, axis=0)
# rmv1_std = np.std(rmv1_arr, axis=0)

# c_qu_std = np.std(c_qu_arr, axis=0)
# cn_qu_std = np.std(cn_qu_arr, axis=0)
# pcn_qu_std = np.std(pcn_qu_arr, axis=0)

# ps_mask_std = np.std(ps_mask_arr, axis=0)
# qu_mask_std = np.std(qu_mask_arr, axis=0)
# inp_qu_std = np.std(inp_qu_arr, axis=0)
inp_eb_std = np.std(inp_eb_arr, axis=0)
# print(f'{rmv_std.shape=}')

# n_list = []
# path_n = glob.glob('./pcn_dl/B/n/*.npy')
# for p in path_n:
#     n = np.load(p)
#     n_list.append(n)

# n_arr = np.array(n_list)
# print(f'{n_arr.shape=}')
# n_mean = np.mean(n_arr, axis=0)
# n_std = np.std(n_arr, axis=0)

plt.figure(1)
# plt.plot(ell_arr, rmv_mean, label='debias rmv_mean')
# plt.plot(ell_arr, rmv1_mean, label='rmv_mean 10sigma')
plt.plot(ell_arr, c_mean, label='c_mean')
plt.plot(ell_arr, cn_mean, label='debias cn_mean')
plt.plot(ell_arr, pcn_mean, label='debias pcn_mean')

# plt.plot(ell_arr, rmv_qu_mean, label='debias rmv_mean qu')
# plt.plot(ell_arr, rmv_lon_lat_mean, label='debias rmv_mean lonlat qu')
# plt.plot(ell_arr, rmv_b_qu_mean, label='debias rmv_mean b on qu')

# plt.plot(ell_arr, c_qu_mean, label='c_mean qu')
# plt.plot(ell_arr, cn_qu_mean, label='debias cn_mean qu')
# plt.plot(ell_arr, pcn_qu_mean, label='debias pcn_mean qu')

# plt.plot(ell_arr, ps_mask_mean, label='debias ps_mask_mean')
# plt.plot(ell_arr, qu_mask_mean, label='debias qu_mask_mean')
# plt.plot(ell_arr, inp_qu_mean, label='debias inp_qu_mean')
plt.plot(ell_arr, inp_eb_mean, label='debias inp_eb_mean')

plt.xlabel('$\\ell$')
plt.ylabel('$D_\\ell^{BB}$')
plt.semilogy()
plt.legend()
plt.title('debiased power spectrum')

plt.figure(2)
# plt.plot(ell_arr, rmv_std, label='rmv_std')
plt.plot(ell_arr, rmv_qu_std, label='rmv_std qu')
plt.plot(ell_arr, rmv_lon_lat_std, label='rmv_std lonlat qu')
# plt.plot(ell_arr, rmv_b_qu_std, label='rmv_std b on qu')
# plt.plot(ell_arr, rmv1_std, label='rmv_std 10sigma')
plt.plot(ell_arr, c_std, label='c_std')
plt.plot(ell_arr, cn_std, label='cn_std')
plt.plot(ell_arr, pcn_std, label='pcn_std')

plt.plot(ell_arr, ps_mask_std, label='ps_mask_std')
plt.plot(ell_arr, qu_mask_std, label='qu_mask_std')
# plt.plot(ell_arr, inp_qu_std, label='inp_qu_std')
plt.plot(ell_arr, inp_eb_std, label='inp_eb_std')
plt.xlabel('$\\ell$')
plt.ylabel('$D_\\ell^{BB}$')
plt.semilogy()
plt.legend()
plt.title('standard deviation')

plt.figure(3)
# plt.plot(ell_arr, rmv_mean - cn_mean, label='rmv res')
plt.plot(ell_arr, rmv_qu_mean - cn_mean, label='rmv qu res')
plt.plot(ell_arr, rmv_lon_lat_mean - cn_mean, label='rmv qu lonlat res')
# plt.plot(ell_arr, rmv_b_qu_mean - cn_mean, label='rmv b on qu res')
plt.plot(ell_arr, pcn_mean - cn_mean, label='pcn res')
plt.plot(ell_arr, ps_mask_mean - cn_mean, label='ps_mask res')
plt.plot(ell_arr, qu_mask_mean - cn_mean, label='qu_mask res')
# plt.plot(ell_arr, inp_qu_mean - cn_mean, label='inp_qu res')
plt.plot(ell_arr, inp_eb_mean - cn_mean, label='inp_eb res')
plt.xlabel('$\\ell$')
plt.ylabel('$D_\\ell^{BB}$')
plt.ylim(-0.1,0.1)
plt.legend()
plt.title('residual power spectrum')

# rmv_rres = (rmv_mean - cn_mean) / cn_mean
# rmv_rres_pos = np.where(rmv_rres > 0, rmv_rres, np.nan)
# rmv_rres_neg = np.where(rmv_rres < 0, np.abs(rmv_rres), np.nan)

pcn_rres = (pcn_mean - cn_mean) / cn_mean
pcn_rres_pos = np.where(pcn_rres > 0, pcn_rres, np.nan)
pcn_rres_neg = np.where(pcn_rres < 0, np.abs(pcn_rres), np.nan)

rmv_qu_rres = (rmv_qu_mean - cn_mean) / cn_mean
rmv_qu_rres_pos = np.where(rmv_qu_rres > 0, rmv_qu_rres, np.nan)
rmv_qu_rres_neg = np.where(rmv_qu_rres < 0, np.abs(rmv_qu_rres), np.nan)

rmv_lon_lat_rres = (rmv_lon_lat_mean - cn_mean) / cn_mean
rmv_lon_lat_rres_pos = np.where(rmv_lon_lat_rres > 0, rmv_lon_lat_rres, np.nan)
rmv_lon_lat_rres_neg = np.where(rmv_lon_lat_rres < 0, np.abs(rmv_lon_lat_rres), np.nan)


# rmv_b_qu_rres = (rmv_b_qu_mean - cn_mean) / cn_mean
# rmv_b_qu_rres_pos = np.where(rmv_b_qu_rres > 0, rmv_b_qu_rres, np.nan)
# rmv_b_qu_rres_neg = np.where(rmv_b_qu_rres < 0, np.abs(rmv_b_qu_rres), np.nan)

# inp_qu_rres = (inp_qu_mean - cn_mean) / cn_mean
# inp_qu_rres_pos = np.where(inp_qu_rres > 0, inp_qu_rres, np.nan)
# inp_qu_rres_neg = np.where(inp_qu_rres < 0, np.abs(inp_qu_rres), np.nan)

ps_mask_rres = (ps_mask_mean - cn_mean) / cn_mean
ps_mask_rres_pos = np.where(ps_mask_rres > 0, ps_mask_rres, np.nan)
ps_mask_rres_neg = np.where(ps_mask_rres < 0, np.abs(ps_mask_rres), np.nan)

qu_mask_rres = (qu_mask_mean - cn_mean) / cn_mean
qu_mask_rres_pos = np.where(qu_mask_rres > 0, qu_mask_rres, np.nan)
qu_mask_rres_neg = np.where(qu_mask_rres < 0, np.abs(qu_mask_rres), np.nan)

inp_eb_rres = (inp_eb_mean - cn_mean) / cn_mean
inp_eb_rres_pos = np.where(inp_eb_rres > 0, inp_eb_rres, np.nan)
inp_eb_rres_neg = np.where(inp_eb_rres < 0, np.abs(inp_eb_rres), np.nan)

plt.figure(4)
plt.scatter(ell_arr, pcn_rres_pos, color='b', marker='+', label='pcn')
plt.scatter(ell_arr, pcn_rres_neg, color='b', marker='_', label='pcn')

# plt.scatter(ell_arr, rmv_rres_pos, color='g', marker='+', label='fit b rmv on b')
# plt.scatter(ell_arr, rmv_rres_neg, color='g', marker='_', label='fit b rmv on b')

plt.scatter(ell_arr, rmv_qu_rres_pos, color='r', marker='+', label='fit qu rmv on qu')
plt.scatter(ell_arr, rmv_qu_rres_neg, color='r', marker='_', label='fit qu rmv on qu')

plt.scatter(ell_arr, rmv_lon_lat_rres_pos, color='purple', marker='+', label='fit lon_lat rmv on qu')
plt.scatter(ell_arr, rmv_lon_lat_rres_neg, color='purple', marker='_', label='fit lon_lat rmv on qu')


# plt.scatter(ell_arr, rmv_b_qu_rres_pos, color='c', marker='+', label='fit b rmv on qu')
# plt.scatter(ell_arr, rmv_b_qu_rres_neg, color='c', marker='_', label='fit b rmv on qu')

plt.scatter(ell_arr, ps_mask_rres_pos, color='m', marker='+', label='ps_mask')
plt.scatter(ell_arr, ps_mask_rres_neg, color='m', marker='_', label='ps_mask')

plt.scatter(ell_arr, qu_mask_rres_pos, color='pink', marker='+', label='qu_mask')
plt.scatter(ell_arr, qu_mask_rres_neg, color='pink', marker='_', label='qu_mask')

# plt.scatter(ell_arr, inp_qu_rres_pos, color='y', marker='+', label='inp_qu')
# plt.scatter(ell_arr, inp_qu_rres_neg, color='y', marker='_', label='inp_qu')

plt.scatter(ell_arr, inp_eb_rres_pos, color='y', marker='+', label='inp_eb')
plt.scatter(ell_arr, inp_eb_rres_neg, color='y', marker='_', label='inp_eb')


plt.scatter(ell_arr, cn_std / cn_mean, color='k', marker='.', label='std')

plt.xlabel('$\\ell$')
plt.ylabel('$D_\\ell^{BB res} / D_\\ell^{BB cn}$')
# plt.ylim(bottom=1e-10)
plt.loglog()
# plt.ylim(-0.1,0.1)
plt.legend(loc='upper right')
plt.title('relative residual power spectrum')

plt.figure(5)
pcn_rmse = np.sqrt(pcn_std**2 + (pcn_mean - cn_mean)**2)
cn_rmse = np.sqrt(cn_std**2)
rmv_qu_rmse = np.sqrt(rmv_qu_std**2 + (rmv_qu_mean - cn_mean)**2)
rmv_lon_lat_rmse = np.sqrt(rmv_lon_lat_std**2 + (rmv_lon_lat_mean - cn_mean)**2)
# rmv_qu_b_rmse = np.sqrt(rmv_b_qu_std**2 + (rmv_b_qu_mean - cn_mean)**2)
inp_eb_rmse = np.sqrt(inp_eb_std**2 + (inp_eb_mean - cn_mean)**2)
# inp_qu_rmse = np.sqrt(inp_qu_std**2 + (inp_qu_mean - cn_mean)**2)
ps_mask_rmse = np.sqrt(ps_mask_std**2 + (ps_mask_mean - cn_mean)**2)
qu_mask_rmse = np.sqrt(qu_mask_std**2 + (qu_mask_mean - cn_mean)**2)

# pcn_rmse_ratio = np.sum(pcn_rmse[1:7] / cn_mean[1:7])
# print(f'{pcn_rmse_ratio=}')
# rmv_rmse_ratio = np.sum(rmv_rmse[1:7] / cn_mean[1:7])
# print(f'{rmv_rmse_ratio=}')
# inp_eb_rmse_ratio = np.sum(inp_eb_rmse[1:7] / cn_mean[1:7])
# print(f'{inp_eb_rmse_ratio=}')
# inp_qu_rmse_ratio = np.sum(inp_qu_rmse[1:7] / cn_mean[1:7])
# print(f'{inp_qu_rmse_ratio=}')
# ps_mask_rmse_ratio = np.sum(ps_mask_rmse[1:7] / cn_mean[1:7])
# print(f'{ps_mask_rmse_ratio=}')

plt.scatter(ell_arr, pcn_rmse, label='pcn ', marker='.')
plt.scatter(ell_arr, cn_rmse, label='cn ', marker='.')
plt.scatter(ell_arr, rmv_qu_rmse, label='rmv qu', marker='.')
plt.scatter(ell_arr, rmv_lon_lat_rmse, label='rmv qu lonlat', marker='.')
# plt.scatter(ell_arr, rmv_qu_rmse, label='rmv b qu', marker='.')
plt.scatter(ell_arr, inp_eb_rmse, label='inp eb ', marker='.')
# plt.scatter(ell_arr, inp_qu_rmse, label='inp qu ', marker='.')
plt.scatter(ell_arr, ps_mask_rmse, label='ps mask ', marker='.')
plt.scatter(ell_arr, qu_mask_rmse, label='qu mask ', marker='.')
plt.xlabel('$\\ell$')
plt.ylabel('$D_\\ell^{BB}$')
# plt.ylim(bottom=1e-10)
plt.loglog()
# plt.ylim(-0.1,0.1)
plt.legend()
plt.title('rmse')

plt.show()





