import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import pymaster as nmt
import glob

from pathlib import Path

lmax = 1999
l = np.arange(lmax+1)
nside = 2048
rlz_idx = 0
threshold = 2

df = pd.read_csv('../../../../FGSim/FreqBand')
freq = df.at[7, 'freq']
beam = df.at[7, 'beam']
print(f'{freq=}, {beam=}')

c_list = []
test_c_list = []

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

l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=1999, fold=0.2)
bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
ell_arr = bin_dl.get_effective_ells()

for rlz_idx in range(1,100):
    if rlz_idx == 50:
        continue
    c = np.load(f'./pcn_dl/B/c/{rlz_idx}.npy')
    print(f'{c.shape=}')
    test_c = np.load(f'./pcn_dl/B/test_c/{rlz_idx}.npy')
    print(f'{test_c.shape=}')


    # plt.plot(ell_arr, c, label=f'c {rlz_idx}')
    # plt.semilogy()
    # plt.legend()

    c_list.append(c)
    test_c_list.append(test_c)

# plt.show()

c_arr = np.array(c_list)
test_c_arr = np.array(test_c_list)

c_mean = np.mean(c_arr, axis=0)
test_c_mean = np.mean(test_c_arr, axis=0)

c_std = np.std(c_arr, axis=0)
test_c_std = np.std(test_c_arr, axis=0)

l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=1999, fold=0.2)
bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
ell_arr = bin_dl.get_effective_ells()

plt.figure(1)
plt.plot(ell_arr, c_mean, label='c_mean')

l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=1200, fold=0.2)
bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
ell_arr = bin_dl.get_effective_ells()

plt.plot(ell_arr, test_c_mean, label='test_c_mean')
plt.xlabel('$\\ell$')
plt.ylabel('$D_\\ell^{BB}$')
plt.semilogy()
plt.legend()
plt.title('debiased power spectrum')

l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=1999, fold=0.2)
bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
ell_arr = bin_dl.get_effective_ells()

plt.figure(2)
plt.plot(ell_arr, c_std, label='c_std')
l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=1200, fold=0.2)
bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
ell_arr = bin_dl.get_effective_ells()
plt.plot(ell_arr, test_c_std, label='test_c_std')

plt.xlabel('$\\ell$')
plt.ylabel('$D_\\ell^{BB}$')
plt.semilogy()
plt.legend()
plt.title('standard deviation')
plt.show()

