import numpy as np
import healpy as hp
import pymaster as nmt
import matplotlib.pyplot as plt

freq = 270
beam = 9
lmax = 1999
l = np.arange(lmax+1)
m = np.load(f'../../../../../fitdata/synthesis_data/2048/CMBNOISE/{freq}/1.npy')
m_q = m[1]
m_u = m[2]

apo_mask = np.load('../../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')
mask_C1_05 = hp.read_map('./3sigma/apo_mask/C1_05/1.fits')
mask_crt_C1_05 = hp.read_map('./3sigma/apo_mask/crt_C1_05/1.fits')

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

cl = hp.anafast(m, lmax=lmax)
bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000)


l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax, fold=0.2)
bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
ell_arr = bin_dl.get_effective_ells()

def test_purify():

    f_2_apo = nmt.NmtField(apo_mask, [m_q, m_u], beam=bl, masked_on_input=False)
    f_2_apo_purify = nmt.NmtField(apo_mask, [m_q, m_u], beam=bl, masked_on_input=False, purify_b=True)
    
    dl_22_apo = nmt.compute_full_master(f_2_apo, f_2_apo, bin_dl)
    dl_22_apo_purify = nmt.compute_full_master(f_2_apo_purify, f_2_apo_purify, bin_dl)
    
    plt.plot(ell_arr, dl_22_apo[0], label='EE')
    plt.plot(ell_arr, dl_22_apo[3], label='BB')
    plt.plot(l, l*(l+1)*cl[2]/(2*np.pi)/bl[0:lmax+1]**2, label='theory')
    plt.semilogy()
    plt.legend()
    plt.show()
    
    plt.plot(ell_arr, dl_22_apo_purify[0], label='EE purify')
    plt.plot(ell_arr, dl_22_apo_purify[3], label='BB purify')
    plt.plot(l, l*(l+1)*cl[2]/(2*np.pi)/bl[0:lmax+1]**2, label='theory')
    plt.semilogy()
    plt.legend()
    plt.show()

f_2_ps = nmt.NmtField(mask=mask_C1_05, maps=[m_q, m_u], beam=bl, masked_on_input=False, purify_b=True)
dl_22_ps = nmt.compute_full_master(f_2_ps, f_2_ps, bin_dl)

f_2_ps_crt = nmt.NmtField(mask=mask_crt_C1_05, maps=[m_q, m_u], beam=bl, masked_on_input=False, purify_b=True)
dl_22_ps_crt = nmt.compute_full_master(f_2_ps_crt, f_2_ps_crt, bin_dl)

plt.plot(ell_arr, dl_22_ps[3], label='BB ps')
plt.plot(ell_arr, dl_22_ps_crt[3], label='BB ps crt')
plt.plot(l, l*(l+1)*cl[2]/(2*np.pi)/bl[0:lmax+1]**2, label='theory')
plt.legend()
plt.semilogy()
plt.show()







