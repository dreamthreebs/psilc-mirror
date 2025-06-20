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

freq = 155
lmax = 1999
beam = 17
nside = 2048
l = np.arange(lmax+1)
bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)

bin_mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
apo_mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')

m_e = np.load('./pcn_after_removal/3sigma/E/map_crp_e1.npy') * bin_mask
m_b = np.load('./pcn_after_removal/3sigma/B/map_cln_b1.npy') * bin_mask

pcn = np.load(f'../../../../fitdata/synthesis_data/2048/PSCMBNOISE/{freq}/1.npy')
_, alm_pcn_e, alm_pcn_b = hp.map2alm(pcn, lmax=lmax)
pcn_e = hp.alm2map(alm_pcn_e, nside=nside)
pcn_b = hp.alm2map(alm_pcn_b, nside=nside)

cn = np.load(f'../../../../fitdata/synthesis_data/2048/CMBNOISE/{freq}/1.npy')
_, alm_cn_e, alm_cn_b = hp.map2alm(cn, lmax=lmax)
cn_e = hp.alm2map(alm_cn_e, nside=nside)
cn_b = hp.alm2map(alm_cn_b, nside=nside)

# inp_qu_e = hp.read_map('./inpaint_pcn/3sigma/QU/E/1.fits')
# inp_qu_b = hp.read_map('./inpaint_pcn/3sigma/QU/B/1.fits')

inp_eb_e = hp.read_map('./inpaint_pcn/3sigma/EB/E_output/1.fits')
inp_eb_b = hp.read_map('./inpaint_pcn/3sigma/EB/B_output/1.fits')

df = pd.read_csv(f'../../../mask/mask_csv/{freq}.csv')

fig_size = 100
e_min = None
e_max = None
b_min = None
b_max = None

flux_idx = 1
lon = np.rad2deg(df.at[flux_idx, 'lon'])
lat = np.rad2deg(df.at[flux_idx, 'lat'])
ctr_vec = hp.ang2vec(theta=lon, phi=lat, lonlat=True)

pix_mask = hp.query_disc(nside=nside, vec=ctr_vec, radius=1.5 * np.deg2rad(beam)/60)
mask = np.ones(hp.nside2npix(nside))
mask[pix_mask] = 0
print(f'{lon=}, {lat=}')

m_proj_cn_E = hp.gnomview(cn_e, rot=[lon, lat, 0], title='cn E', xsize=fig_size, return_projected_map=True)
e_min = 1.5 * np.min(m_proj_cn_E)
e_max = 1.5 * np.max(m_proj_cn_E)
plt.show()
m_proj_cn_B = hp.gnomview(cn_b, rot=[lon, lat, 0], title='cn B', xsize=fig_size, return_projected_map=True)
b_min = 1.5 * np.min(m_proj_cn_B)
b_max = 1.5 * np.max(m_proj_cn_B)
plt.show()

plt.figure(1)
hp.gnomview(pcn_e, rot=[lon, lat, 0], title='pcn e', xsize=fig_size, ysize=fig_size, min=e_min, max=e_max, sub=231)
hp.gnomview(cn_e, rot=[lon, lat, 0], title='cn e', xsize=fig_size, ysize=fig_size, min=e_min, max=e_max, sub=232)
# hp.gnomview(inp_qu_e, rot=[lon, lat, 0], title='inp qu e', xsize=fig_size, ysize=fig_size, min=e_min, max=e_max, sub=243)
hp.gnomview(inp_eb_e, rot=[lon, lat, 0], title='inp eb e', xsize=fig_size, ysize=fig_size, min=e_min, max=e_max, sub=233)
hp.gnomview(m_e, rot=[lon, lat, 0], title='removal e', xsize=fig_size, ysize=fig_size, min=e_min, max=e_max, sub=234)
hp.gnomview(m_e - cn_e, rot=[lon, lat, 0], title='residual removal e', xsize=fig_size, ysize=fig_size, min=e_min, max=e_max, sub=235)
# hp.gnomview(inp_qu_e - cn_e, rot=[lon, lat, 0], title='residual inpaint qu e', xsize=fig_size, ysize=fig_size, min=e_min, max=e_max, sub=247)
hp.gnomview(inp_eb_e - cn_e, rot=[lon, lat, 0], title='residual inpaint eb e', xsize=fig_size, ysize=fig_size, min=e_min, max=e_max, sub=236)

plt.figure(2)
hp.gnomview(pcn_b, rot=[lon, lat, 0], title='pcn b', xsize=fig_size, ysize=fig_size, min=b_min, max=b_max, sub=231)
hp.gnomview(cn_b, rot=[lon, lat, 0], title='cn b', xsize=fig_size, ysize=fig_size, min=b_min, max=b_max, sub=232)
# hp.gnomview(inp_qu_b, rot=[lon, lat, 0], title='inp qu b', xsize=fig_size, ysize=fig_size, min=b_min, max=b_max, sub=243)
hp.gnomview(inp_eb_b, rot=[lon, lat, 0], title='inp eb b', xsize=fig_size, ysize=fig_size, min=b_min, max=b_max, sub=233)
hp.gnomview(m_b, rot=[lon, lat, 0], title='removal b', xsize=fig_size, ysize=fig_size, min=b_min, max=b_max, sub=234)
hp.gnomview(m_b - cn_b, rot=[lon, lat, 0], title='residual removal b', xsize=fig_size, ysize=fig_size, min=b_min, max=b_max, sub=235)
# hp.gnomview(inp_qu_b - cn_b, rot=[lon, lat, 0], title='residual inpaint qu b', xsize=fig_size, ysize=fig_size, min=b_min, max=b_max, sub=247)
hp.gnomview(inp_eb_b - cn_b, rot=[lon, lat, 0], title='residual inpaint eb b', xsize=fig_size, ysize=fig_size, min=b_min, max=b_max, sub=236)

plt.figure(3)
hp.gnomview(pcn_e * mask, rot=[lon, lat, 0], title='pcn e masked', xsize=fig_size, ysize=fig_size, min=e_min, max=e_max, sub=221)
hp.gnomview(pcn_b * mask, rot=[lon, lat, 0], title='pcn b masked', xsize=fig_size, ysize=fig_size, min=b_min, max=b_max, sub=222)
hp.gnomview(cn_e, rot=[lon, lat, 0], title='cn e', xsize=fig_size, ysize=fig_size, min=e_min, max=e_max, sub=223)
hp.gnomview(cn_b, rot=[lon, lat, 0], title='cn b', xsize=fig_size, ysize=fig_size, min=b_min, max=b_max, sub=224)

plt.show()




