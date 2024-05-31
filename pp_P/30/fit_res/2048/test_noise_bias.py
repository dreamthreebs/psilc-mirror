import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import pymaster as nmt
import glob

from pathlib import Path

lmax = 400
l = np.arange(lmax+1)
nside = 2048
rlz_idx = 10
threshold = 2

df = pd.read_csv('../../../../FGSim/FreqBand')
freq = df.at[0, 'freq']
beam = df.at[0, 'beam']
print(f'{freq=}, {beam=}')

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

c = np.load(f'./pcn_dl/B/c/{rlz_idx}.npy')
cn = np.load(f'./pcn_dl/B/cn/{rlz_idx}.npy')
n = np.load(f'./pcn_dl/B/n_true/{rlz_idx}.npy')

n_list = []
path_n = glob.glob('./pcn_dl/B/n/*.npy')

for p in path_n:
    n = np.load(p)
    n_list.append(n)

n_arr = np.array(n_list)
print(f'{n_arr.shape=}')
n_mean = np.mean(n_arr[0:10,:], axis=0)


plt.plot(ell_arr, c, label='c')
plt.plot(ell_arr, cn, label='cn')
plt.plot(ell_arr, n, label='n')
plt.plot(ell_arr, n_mean, label='n mean from 1000 noise realization')
plt.plot(ell_arr, np.abs(cn-n_mean), label='debias mean noise realization')
plt.plot(ell_arr, np.abs(cn-n), label='debias noise cn')

plt.xlabel('$\\ell$')
plt.ylabel('$D_\\ell^{BB}$')
plt.semilogy()
plt.legend()
plt.show()

