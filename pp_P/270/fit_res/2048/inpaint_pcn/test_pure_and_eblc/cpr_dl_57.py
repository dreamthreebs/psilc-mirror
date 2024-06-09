import numpy as np
import healpy as hp
import pymaster as nmt
import matplotlib.pyplot as plt

from pathlib import Path
from eblc_base import EBLeakageCorrection

beam = 9
lmax = 1500
nside = 512
l = np.arange(lmax+1)
rlz_idx = 57

m = np.load(f'./cn/{rlz_idx}.npy')
m_q = m[1]
m_u = m[2]

# apo_mask = np.load('./mask/apo_edge_mask.npy')
apo_mask = np.load('./mask/final_mask.npy')
bin_mask = np.load('./mask/final_bin_mask.npy')

# hp.mollview(m[0])
# hp.mollview(m[1])
# hp.mollview(m[2])
# plt.show()

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
l_min_edges, l_max_edges = generate_bins(l_min_start=20, delta_l_min=20, l_max=lmax, fold=0.2)
bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
ell_arr = bin_dl.get_effective_ells()

def pure_b():
    f_2_apo = nmt.NmtField(apo_mask, [m_q, m_u], masked_on_input=False)
    f_2_apo_purify = nmt.NmtField(apo_mask, [m_q, m_u], masked_on_input=False, purify_b=True)
    
    dl_22_apo = nmt.compute_full_master(f_2_apo, f_2_apo, bin_dl)
    dl_22_apo_purify = nmt.compute_full_master(f_2_apo_purify, f_2_apo_purify, bin_dl)
    return dl_22_apo, dl_22_apo_purify
    
def eblc():
    obj_eblc = EBLeakageCorrection(m=m, lmax=lmax, nside=nside, mask=bin_mask, post_mask=bin_mask)
    _,_,cln_b = obj_eblc.run_eblc()
    f1 = nmt.NmtField(mask=apo_mask, maps=[cln_b], masked_on_input=False)
    eblc_dl = nmt.compute_full_master(f1, f1, bin_dl)
    return eblc_dl[0]

def cal_e():
    crt_e = hp.alm2map(hp.map2alm(m*bin_mask, lmax=lmax)[1], nside=nside)
    f1 = nmt.NmtField(mask=apo_mask, maps=[crt_e], masked_on_input=False)
    e_dl = nmt.compute_full_master(f1, f1, bin_dl)
    return e_dl[0]

def no_lkg_b():
    full_b = hp.alm2map(hp.map2alm(m, lmax=lmax)[2], nside=nside)
    f1 = nmt.NmtField(mask=apo_mask, maps=[full_b], masked_on_input=False)
    std_dl_b = nmt.compute_full_master(f1, f1, bin_dl)
    return std_dl_b[0]


# dl_22_apo, dl_22_apo_purify = pure_b()
# dl_eblc = eblc()
# dl_e = cal_e()

dl_b_no_lkg = no_lkg_b()

# plt.plot(ell_arr, dl_eblc, label='BB')
# plt.plot(ell_arr, dl_e, label='EE from E map')
# plt.plot(ell_arr, dl_22_apo[0], label='EE')
# plt.plot(ell_arr, dl_22_apo[3], label='BB no purify')
# plt.plot(ell_arr, dl_22_apo_purify[0], label='EE purify')
# plt.plot(ell_arr, dl_22_apo_purify[3], label='BB purify')
# plt.plot(l, l*(l+1)*cl[2]/(2*np.pi), label='theory')
# plt.semilogy()
# plt.legend()
# plt.show()

# path_dl_cln_b = Path('./dl_data/cln_b')
# path_dl_crt_e = Path('./dl_data/crt_e')
# path_dl_no_pure_b = Path('./dl_data/no_pure_b')
# path_dl_no_pure_e = Path('./dl_data/no_pure_e')
# path_dl_pure_b = Path('./dl_data/pure_b')

path_dl_b_no_lkg = Path('./dl_data/no_lkg')

# path_dl_cln_b.mkdir(exist_ok=True, parents=True)
# path_dl_crt_e.mkdir(exist_ok=True, parents=True)
# path_dl_no_pure_b.mkdir(exist_ok=True, parents=True)
# path_dl_no_pure_e.mkdir(exist_ok=True, parents=True)
# path_dl_pure_b.mkdir(exist_ok=True, parents=True)

path_dl_b_no_lkg.mkdir(exist_ok=True, parents=True)

# np.save(path_dl_cln_b / Path(f'{rlz_idx}.npy'), dl_eblc)
# np.save(path_dl_crt_e / Path(f'{rlz_idx}.npy'), dl_e)
# np.save(path_dl_no_pure_b / Path(f'{rlz_idx}.npy'), dl_22_apo[3])
# np.save(path_dl_no_pure_e / Path(f'{rlz_idx}.npy'), dl_22_apo[0])
# np.save(path_dl_pure_b / Path(f'{rlz_idx}.npy'), dl_22_apo_purify[3])

np.save(path_dl_b_no_lkg / Path(f'{rlz_idx}.npy'), dl_b_no_lkg)

# np.save('./dl_data/ell_arr.npy', ell_arr)
# np.save('./dl_data/dl_theory.npy', l*(l+1)*cl/(2*np.pi))



