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

freq = 270
lmax = 1999
beam = 9
nside = 2048
l = np.arange(lmax+1)
bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)

l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax, fold=0.2)
bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
ell_arr = bin_dl.get_effective_ells()

bin_mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
apo_mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')

m_q = np.load('./pcfn_after_removal/3sigma/map_q_1.npy') * bin_mask
m_u = np.load('./pcfn_after_removal/3sigma/map_u_1.npy') * bin_mask

pcfn = np.load(f'../../../../fitdata/synthesis_data/2048/PSCMBFGNOISE/{freq}/1.npy') * bin_mask
pcfn_q = pcfn[1].copy()
pcfn_u = pcfn[2].copy()

cfn = np.load(f'../../../../fitdata/synthesis_data/2048/CMBFGNOISE/{freq}/1.npy') * bin_mask
cfn_q = cfn[1].copy()
cfn_u = cfn[2].copy()

df = pd.read_csv(f'../../../mask/mask_csv/{freq}.csv')

fig_size = 30
q_min = -15
q_max = 15
u_min = -15
u_max = 15

flux_idx = 1
lon = np.rad2deg(df.at[flux_idx, 'lon'])
lat = np.rad2deg(df.at[flux_idx, 'lat'])
ctr_vec = hp.ang2vec(theta=lon, phi=lat, lonlat=True)

pix_mask = hp.query_disc(nside=nside, vec=ctr_vec, radius=1.5 * np.deg2rad(beam)/60)
mask = np.ones(hp.nside2npix(nside))
mask[pix_mask] = 0
print(f'{lon=}, {lat=}')

plt.figure(1)
hp.gnomview(pcfn_q, rot=[lon, lat, 0], title='pcfn q', xsize=fig_size, ysize=fig_size, min=q_min, max=q_max, sub=221)
hp.gnomview(cfn_q, rot=[lon, lat, 0], title='cfn q', xsize=fig_size, ysize=fig_size, min=q_min, max=q_max, sub=222)
hp.gnomview(m_q, rot=[lon, lat, 0], title='removal q', xsize=fig_size, ysize=fig_size, min=q_min, max=q_max, sub=223)
hp.gnomview(m_q - cfn_q, rot=[lon, lat, 0], title='residual q', xsize=fig_size, ysize=fig_size, sub=224)

plt.figure(2)
hp.gnomview(pcfn_u, rot=[lon, lat, 0], title='pcfn u', xsize=fig_size, ysize=fig_size, min=u_min, max=u_max, sub=221)
hp.gnomview(cfn_u, rot=[lon, lat, 0], title='cfn u', xsize=fig_size, ysize=fig_size, min=u_min, max=u_max, sub=222)
hp.gnomview(m_u, rot=[lon, lat, 0], title='removal u', xsize=fig_size, ysize=fig_size, min=u_min, max=u_max, sub=223)
hp.gnomview(m_u - cfn_u, rot=[lon, lat, 0], title='residual u', xsize=fig_size, ysize=fig_size, sub=224)

plt.figure(3)
hp.gnomview(pcfn_q * mask, rot=[lon, lat, 0], title='pcfn q masked', xsize=fig_size, ysize=fig_size, min=q_min, max=q_max, sub=221)
hp.gnomview(pcfn_u * mask, rot=[lon, lat, 0], title='pcfn u masked', xsize=fig_size, ysize=fig_size, min=q_min, max=q_max, sub=222)
hp.gnomview(cfn_q, rot=[lon, lat, 0], title='cfn q', xsize=fig_size, ysize=fig_size, min=q_min, max=q_max, sub=223)
hp.gnomview(cfn_u, rot=[lon, lat, 0], title='cfn u', xsize=fig_size, ysize=fig_size, min=q_min, max=q_max, sub=224)

plt.figure(4)
hp.orthview(pcfn_q, rot=[100,50,0], title='pcfn q', sub=251, half_sky=True, xsize=2000, min=-1, max=1)
hp.orthview(pcfn_u, rot=[100,50,0], title='pcfn u', sub=252, half_sky=True, xsize=2000, min=-1, max=1)
hp.orthview(cfn_q, rot=[100,50,0], title='cfn q', sub=253, half_sky=True, xsize=2000, min=-1, max=1)
hp.orthview(cfn_u, rot=[100,50,0], title='cfn u', sub=254, half_sky=True, xsize=2000, min=-1, max=1)
hp.orthview(m_q, rot=[100,50,0], title='rmv q', sub=255, half_sky=True, xsize=2000, min=-1, max=1)
hp.orthview(m_u, rot=[100,50,0], title='rmv u', sub=256, half_sky=True, xsize=2000, min=-1, max=1)
hp.orthview(m_q - cfn_q, rot=[100,50,0], title='res rmv q', sub=257, half_sky=True, xsize=2000, min=-1, max=1)
hp.orthview(m_u - cfn_u, rot=[100,50,0], title='res rmv u', sub=258, half_sky=True, xsize=2000, min=-1, max=1)
hp.orthview(pcfn_q - cfn_q, rot=[100,50,0], title='res pcfn q', sub=259, half_sky=True, xsize=2000, min=-1, max=1)
hp.orthview(pcfn_u - cfn_u, rot=[100,50,0], title='res pcfn u', sub=(2,5,10), half_sky=True, xsize=2000, min=-1, max=1)




plt.show()


