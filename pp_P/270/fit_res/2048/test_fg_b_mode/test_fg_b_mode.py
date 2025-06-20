import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt

from eblc_base import EBLeakageCorrection

beam = 9
lmax = 1999
nside = 2048
# m = np.load('../../../../fitdata/2048/FG/270/fg.npy')
# m = np.load('../../../../../fitdata/synthesis_data/2048/CMBFGNOISE/270/1.npy')
m = np.load('../../../../../fitdata/2048/CMB/270/1.npy')
mask = np.load('../../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
apo_mask = np.load('../../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')

def gen_b_map():
    full_b = hp.alm2map(hp.map2alm(m, lmax=lmax)[2], nside=nside) * mask
    alm_b = hp.map2alm(m*mask, lmax=lmax)[2]
    cut_b = hp.alm2map(alm_b, nside=nside) * mask
    cut_qu = hp.alm2map([np.zeros_like(alm_b), np.zeros_like(alm_b), alm_b], nside=nside)
    cut_qu_b = hp.alm2map(hp.map2alm(cut_qu * mask, lmax=lmax)[2], nside=nside) * mask
    # cut_full_qu_b = hp.alm2map(hp.map2alm(cut_qu, lmax=lmax)[2], nside=nside) * mask
    return cut_b, cut_qu_b, full_b

def get_map():
    obj_eblc = EBLeakageCorrection(m, lmax, nside, mask=mask, post_mask=mask)
    _,_,cln_b = obj_eblc.run_eblc()
    np.save('./cln_b.npy', cln_b)
    cut_b, cut_qu_b, full_b = gen_b_map()
    np.save('./cut_b.npy', cut_b)
    np.save('./cut_qu_b.npy', cut_qu_b)
    np.save('./full_b.npy', full_b)

def check_map():
    hp.orthview(full_b, rot=[100,50,0], title='full_b', half_sky=True)
    hp.orthview(cut_b, rot=[100,50,0], title='cut_b', half_sky=True)
    hp.orthview(cut_qu_b, rot=[100,50,0], title='cut_qu_b', half_sky=True)
    hp.orthview(cln_b, rot=[100,50,0], title='cln_b', half_sky=True)
    # hp.orthview(cut_full_qu_b, rot=[100,50,0], title='full_qu_b', half_sky=True)
    plt.show()

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


def main():
    # get_map()
    cut_b = np.load('./cfn/cut_b.npy')
    cut_qu_b = np.load('./cfn/cut_qu_b.npy')
    full_b = np.load('./cfn/full_b.npy')
    cln_b = np.load('./cfn/cln_b.npy')

    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,1]
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax, fold=0.2)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    dl_cut_b = calc_dl_from_scalar_map(cut_b, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)
    dl_cut_qu_b = calc_dl_from_scalar_map(cut_qu_b, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)
    dl_full_b = calc_dl_from_scalar_map(full_b, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)
    dl_cln_b = calc_dl_from_scalar_map(cln_b, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)

    plt.loglog(ell_arr, dl_cut_b, label='cut_b')
    plt.loglog(ell_arr, dl_cut_qu_b, label='cut_qu_b')
    plt.loglog(ell_arr, dl_full_b, label='full_b')
    plt.loglog(ell_arr, dl_cln_b, label='cln_b')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()



