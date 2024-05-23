import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import pymaster as nmt

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

def calc_dl_from_scalar_map(scalar_map, bl, apo_mask, bin_dl, masked_on_input):
    scalar_field = nmt.NmtField(apo_mask, [scalar_map], beam=bl, masked_on_input=masked_on_input)
    dl = nmt.compute_full_master(scalar_field, scalar_field, bin_dl)
    return dl[0]

freq = 145
lmax = 1999
beam = 19
l = np.arange(lmax+1)
bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)

l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax, fold=0.2)
bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
ell_arr = bin_dl.get_effective_ells()

bin_mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
apo_mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')

m_q = np.load('./pcn_after_removal/2sigma/map_q_1.npy') * bin_mask
m_u = np.load('./pcn_after_removal/2sigma/map_u_1.npy') * bin_mask

pcn = np.load(f'../../../../fitdata/synthesis_data/2048/PSCMBNOISE/{freq}/1.npy') * bin_mask
pcn_q = pcn[1].copy()
pcn_u = pcn[2].copy()

cn = np.load(f'../../../../fitdata/synthesis_data/2048/CMBNOISE/{freq}/1.npy') * bin_mask
cn_q = cn[1].copy()
cn_u = cn[2].copy()

df = pd.read_csv(f'../../../mask/mask_csv/{freq}.csv')

fig_size = 60
q_min = -7
q_max = 7
u_min = -7
u_max = 7

flux_idx = 1
lon = np.rad2deg(df.at[flux_idx, 'lon'])
lat = np.rad2deg(df.at[flux_idx, 'lat'])
print(f'{lon=}, {lat=}')

plt.figure(1)
hp.gnomview(pcn_q, rot=[lon, lat, 0], title='pcn q', xsize=fig_size, ysize=fig_size, min=q_min, max=q_max, sub=221)
hp.gnomview(cn_q, rot=[lon, lat, 0], title='cn q', xsize=fig_size, ysize=fig_size, min=q_min, max=q_max, sub=222)
hp.gnomview(m_q, rot=[lon, lat, 0], title='removal q', xsize=fig_size, ysize=fig_size, min=q_min, max=q_max, sub=223)
hp.gnomview(m_q - cn_q, rot=[lon, lat, 0], title='residual q', xsize=fig_size, ysize=fig_size, sub=224)
plt.figure(2)

hp.gnomview(pcn_u, rot=[lon, lat, 0], title='pcn u', xsize=fig_size, ysize=fig_size, min=u_min, max=u_max, sub=221)
hp.gnomview(cn_u, rot=[lon, lat, 0], title='cn u', xsize=fig_size, ysize=fig_size, min=u_min, max=u_max, sub=222)
hp.gnomview(m_u, rot=[lon, lat, 0], title='removal u', xsize=fig_size, ysize=fig_size, min=u_min, max=u_max, sub=223)
hp.gnomview(m_u - cn_u, rot=[lon, lat, 0], title='residual u', xsize=fig_size, ysize=fig_size, sub=224)

plt.show()



