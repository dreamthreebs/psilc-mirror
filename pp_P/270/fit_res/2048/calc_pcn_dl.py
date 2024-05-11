import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import pymaster as nmt

from pathlib import Path

lmax = 1999
l = np.arange(lmax+1)
nside = 2048
rlz_idx = 0
threshold = 5

df = pd.read_csv('../../../../FGSim/FreqBand')
freq = df.at[7, 'freq']
beam = df.at[7, 'beam']
print(f'{freq=}, {beam=}')

bin_mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
apo_mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')

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

def cpr_spectrum_pcn_b(bin_mask, apo_mask):

    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=1999, fold=0.2)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    # m_c = np.load(f'../../../../fitdata/2048/CMB/{freq}/{rlz_idx}.npy')
    # m_cn = np.load(f'../../../../fitdata/synthesis_data/2048/CMBNOISE/{freq}/{rlz_idx}.npy')
    # m_pcn = np.load(f'../../../../fitdata/synthesis_data/2048/PSCMBNOISE/{freq}/{rlz_idx}.npy')

    # m_c_b = hp.alm2map(hp.map2alm(m_c, lmax=lmax)[2], nside=nside) * bin_mask
    # m_cn_b = hp.alm2map(hp.map2alm(m_cn, lmax=lmax)[2], nside=nside) * bin_mask
    # m_pcn_b = hp.alm2map(hp.map2alm(m_pcn, lmax=lmax)[2], nside=nside) * bin_mask

    m_removal_b = np.load(f'./pcn_after_removal/{threshold}sigma/B/map_cln_b{rlz_idx}.npy')

    # ### test: checking map
    # hp.orthview(m_cn_b, rot=[100,50,0], half_sky=True, title=' cn b ')
    # hp.orthview(m_pcn_b, rot=[100,50,0], half_sky=True, title=' pcn b ')
    # hp.orthview(m_removal_b, rot=[100,50,0], half_sky=True, title=' removal b ')
    # plt.show()

    # dl_c_b = calc_dl_from_scalar_map(m_c_b, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)
    # dl_cn_b = calc_dl_from_scalar_map(m_cn_b, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)
    # dl_pcn_b = calc_dl_from_scalar_map(m_pcn_b, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)

    dl_removal_b = calc_dl_from_scalar_map(m_removal_b, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)

    path_dl_c = Path(f'pcn_dl/B/c')
    path_dl_c.mkdir(parents=True, exist_ok=True)
    path_dl_cn = Path(f'pcn_dl/B/cn')
    path_dl_cn.mkdir(parents=True, exist_ok=True)
    path_dl_pcn = Path(f'pcn_dl/B/pcn')
    path_dl_pcn.mkdir(parents=True, exist_ok=True)
    path_dl_removal = Path(f'pcn_dl/B/removal_{threshold}sigma')
    path_dl_removal.mkdir(parents=True, exist_ok=True)
    path_dl_inpaint = Path(f'pcn_dl/B/inpaint_{threshold}sigma')
    path_dl_inpaint.mkdir(parents=True, exist_ok=True)

    # np.save(path_dl_c / Path(f'{rlz_idx}.npy'), dl_c_b)
    # np.save(path_dl_cn / Path(f'{rlz_idx}.npy'), dl_cn_b)
    # np.save(path_dl_pcn / Path(f'{rlz_idx}.npy'), dl_pcn_b)

    np.save(path_dl_removal / Path(f'{rlz_idx}.npy'), dl_removal_b)

    # plt.plot(ell_arr, dl_c_b, label='c b', marker='o')
    # plt.plot(ell_arr, dl_cn_b, label='cn b', marker='o')
    # plt.plot(ell_arr, dl_pcn_b, label='pcn b', marker='o')
    # plt.plot(ell_arr, dl_removal_b, label='removal b', marker='o')
    # plt.semilogy()
    # plt.xlabel('$\\ell$')
    # plt.ylabel('$D_\\ell$')
    # plt.legend()
    # plt.show()

def cpr_spectrum_pcn_e(bin_mask, apo_mask):

    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,1]
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=1999, fold=0.2)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    # m_c = np.load(f'../../../../fitdata/2048/CMB/{freq}/{rlz_idx}.npy')
    # m_cn = np.load(f'../../../../fitdata/synthesis_data/2048/CMBNOISE/{freq}/{rlz_idx}.npy')
    # m_pcn = np.load(f'../../../../fitdata/synthesis_data/2048/PSCMBNOISE/{freq}/{rlz_idx}.npy')

    # m_c_e = hp.alm2map(hp.map2alm(m_c, lmax=lmax)[1], nside=nside) * bin_mask
    # m_cn_e = hp.alm2map(hp.map2alm(m_cn, lmax=lmax)[1], nside=nside) * bin_mask
    # m_pcn_e = hp.alm2map(hp.map2alm(m_pcn, lmax=lmax)[1], nside=nside) * bin_mask

    m_removal_e = np.load(f'./pcn_after_removal/{threshold}sigma/E/map_crp_e{rlz_idx}.npy')

    # ### test: checking map
    # hp.orthview(m_cn_e, rot=[100,50,0], half_sky=True, title=' cn e ')
    # hp.orthview(m_pcn_e, rot=[100,50,0], half_sky=True, title=' pcn e ')
    # hp.orthview(m_removal_e, rot=[100,50,0], half_sky=True, title=' removal e ')
    # plt.show()

    # dl_c_e = calc_dl_from_scalar_map(m_c_e, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)
    # dl_cn_e = calc_dl_from_scalar_map(m_cn_e, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)
    # dl_pcn_e = calc_dl_from_scalar_map(m_pcn_e, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)

    dl_removal_e = calc_dl_from_scalar_map(m_removal_e, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)

    path_dl_c = Path(f'pcn_dl/E/c')
    path_dl_c.mkdir(parents=True, exist_ok=True)
    path_dl_cn = Path(f'pcn_dl/E/cn')
    path_dl_cn.mkdir(parents=True, exist_ok=True)
    path_dl_pcn = Path(f'pcn_dl/E/pcn')
    path_dl_pcn.mkdir(parents=True, exist_ok=True)
    path_dl_removal = Path(f'pcn_dl/E/removal_{threshold}sigma')
    path_dl_removal.mkdir(parents=True, exist_ok=True)
    path_dl_inpaint = Path(f'pcn_dl/E/inpaint_{threshold}sigma')
    path_dl_inpaint.mkdir(parents=True, exist_ok=True)

    # np.save(path_dl_c / Path(f'{rlz_idx}.npy'), dl_c_e)
    # np.save(path_dl_cn / Path(f'{rlz_idx}.npy'), dl_cn_e)
    # np.save(path_dl_pcn / Path(f'{rlz_idx}.npy'), dl_pcn_e)

    np.save(path_dl_removal / Path(f'{rlz_idx}.npy'), dl_removal_e)

    # plt.plot(ell_arr, dl_c_e, label='c e', marker='o')
    # plt.plot(ell_arr, dl_cn_e, label='cn e', marker='o')
    # plt.plot(ell_arr, dl_pcn_e, label='pcn e', marker='o')
    # plt.plot(ell_arr, dl_removal_e, label='removal e', marker='o')
    # plt.semilogy()
    # plt.xlabel('$\\ell$')
    # plt.ylabel('$D_\\ell$')
    # plt.legend()
    # plt.show()


def main():
    cpr_spectrum_pcn_b(bin_mask=bin_mask, apo_mask=apo_mask)
    # cpr_spectrum_pcn_e(bin_mask=bin_mask, apo_mask=apo_mask)

main()


