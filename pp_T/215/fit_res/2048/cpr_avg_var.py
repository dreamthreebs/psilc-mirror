import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import pymaster as nmt
from pathlib import Path

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

def gen_ps_remove_map_pcn(rlz_idx, mask, m_cmb_noise):

    m_res = np.load(f'./ps_cmb_noise_residual/2sigma/map{rlz_idx}.npy')
    m_all = m_res + m_cmb_noise
    
    # hp.orthview(m_res * mask, rot=[100,50,0], half_sky=True)
    # plt.show()
    return m_all * mask

def gen_ps_remove_map_pcfn(rlz_idx, mask):

    m_res = np.load('./ps_cmb_noise_residual/2sigma/map0.npy') # TODO
    m_cmb_noise = np.load('../../../../fitdata/synthesis_data/2048/CMBNOISE/155/0.npy')[0].copy()
    m_all = m_res + m_cmb_noise
    
    # hp.orthview(m_all * mask, rot=[100,50,0], half_sky=True)
    # plt.show()
    return m_all * mask

def gen_inpaint_res_map_pcn(rlz_idx, mask, m_cmb_noise):

    m_inpaint = hp.read_map(f'./INPAINT/output/pcn/2sigma/{rlz_idx}.fits', field=0)
    m_res = m_inpaint - m_cmb_noise
    
    # hp.orthview(m_res * mask, rot=[100,50,0], half_sky=True)
    # plt.show()
    return m_res * mask


def main():
    lmax = 2000
    l = np.arange(lmax+1)
    nside = 2048
    df = pd.read_csv('../../../../FGSim/FreqBand')

    freq = df.at[6, 'freq']
    beam = df.at[6, 'beam']
    print(f'{freq=}, {beam=}')
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,0]

    bin_mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
    apo_mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')

    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=2000, fold=0.2)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    dl_inpaint_list = []
    dl_removal_list = []
    dl_pcn_list = []
    dl_cn_list = []
    for rlz_idx in range(100):
        print(f'{rlz_idx=}')

        m_cn = np.load(f'../../../../fitdata/synthesis_data/2048/CMBNOISE/{freq}/{rlz_idx}.npy')[0].copy()
        m_pcn = np.load(f'../../../../fitdata/synthesis_data/2048/PSCMBNOISE/{freq}/{rlz_idx}.npy')[0].copy()
        m_inpaint = hp.read_map(f'./INPAINT/output/pcn/2sigma/{rlz_idx}.fits', field=0) * bin_mask * apo_mask
        m_removal = gen_ps_remove_map_pcn(rlz_idx=rlz_idx, mask=bin_mask, m_cmb_noise=m_cn) * apo_mask

        dl_cn = calc_dl_from_scalar_map(m_cn, bl, apo_mask, bin_dl, masked_on_input=False)
        dl_pcn = calc_dl_from_scalar_map(m_pcn, bl, apo_mask, bin_dl, masked_on_input=False)
        dl_inpaint = calc_dl_from_scalar_map(m_inpaint, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=True)
        dl_removal = calc_dl_from_scalar_map(m_removal, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=True)

        # plt.plot(ell_arr, dl_pcn, label='ps cmb noise', marker='o')
        # plt.plot(ell_arr, dl_cn, label='cmb noise', marker='o')
        # plt.plot(ell_arr, dl_inpaint, label='inpaint', marker='o')
        # plt.plot(ell_arr, dl_removal, label='removal', marker='o')
        # plt.xlabel('$\\ell$')
        # # plt.ylabel('$\\Delta D_\\ell^{TT} [\\mu K^2]$')
        # plt.ylabel('$D_\\ell^{TT} [\\mu K^2]$')
        # plt.legend()
        # # plt.savefig('./fig/pcn/cpr_ps.png', dpi=300)
        # plt.show()

        dl_inpaint_list.append(dl_inpaint)
        dl_removal_list.append(dl_removal)
        dl_pcn_list.append(dl_pcn)
        dl_cn_list.append(dl_cn)

    dl_inpaint_arr = np.asarray(dl_inpaint_list)
    dl_removal_arr = np.asarray(dl_removal_list)
    dl_pcn_arr = np.asarray(dl_pcn_list)
    dl_cn_arr = np.asarray(dl_cn_list)

    dl_inpaint_avg = np.mean(dl_inpaint_arr, axis=0)
    dl_inpaint_var = np.var(dl_inpaint_arr, axis=0)

    dl_removal_avg = np.mean(dl_removal_arr, axis=0)
    dl_removal_var = np.var(dl_removal_arr, axis=0)

    dl_pcn_avg = np.mean(dl_pcn_arr, axis=0)
    dl_pcn_var = np.var(dl_pcn_arr, axis=0)

    dl_cn_avg = np.mean(dl_cn_arr, axis=0)
    dl_cn_var = np.var(dl_cn_arr, axis=0)

    path_avg_var = Path(f'./pcn_avg_var')
    path_avg_var.mkdir(exist_ok=True, parents=True)

    np.save(path_avg_var / Path(f'inpaint_avg.npy'), dl_inpaint_avg)
    np.save(path_avg_var / Path(f'inpaint_var.npy'), dl_inpaint_var)

    np.save(path_avg_var / Path(f'removal_avg.npy'), dl_removal_avg)
    np.save(path_avg_var / Path(f'removal_var.npy'), dl_removal_var)

    np.save(path_avg_var / Path(f'pcn_avg.npy'), dl_pcn_avg)
    np.save(path_avg_var / Path(f'pcn_var.npy'), dl_pcn_var)

    np.save(path_avg_var / Path(f'cn_avg.npy'), dl_cn_avg)
    np.save(path_avg_var / Path(f'cn_var.npy'), dl_cn_var)

def plot_dl():
    lmax = 2000
    l = np.arange(lmax+1)
    nside = 2048
    df = pd.read_csv('../../../../FGSim/FreqBand')

    freq = df.at[6, 'freq']
    beam = df.at[6, 'beam']
    print(f'{freq=}, {beam=}')
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,0]

    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=2000, fold=0.2)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    bin_mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
    apo_mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')

    pcn_avg = np.load('./pcn_avg_var/pcn_avg.npy')
    pcn_var = np.load('./pcn_avg_var/pcn_var.npy')

    cn_avg = np.load(f'./pcn_avg_var/cn_avg.npy')
    cn_var = np.load(f'./pcn_avg_var/cn_var.npy')

    inpaint_avg = np.load(f'./pcn_avg_var/inpaint_avg.npy')
    inpaint_var = np.load(f'./pcn_avg_var/inpaint_var.npy')

    removal_avg = np.load(f'./pcn_avg_var/removal_avg.npy')
    removal_var = np.load(f'./pcn_avg_var/removal_var.npy')

    delta_ps = pcn_avg - cn_avg
    delta_inpaint = inpaint_avg - cn_avg
    delta_removal = removal_avg - cn_avg

    plt.figure(1)
    plt.plot(ell_arr, pcn_avg, label='pcn_avg')
    plt.plot(ell_arr, cn_avg, label='cn_avg')
    plt.plot(ell_arr, inpaint_avg, label='inpaint_avg')
    plt.plot(ell_arr, removal_avg, label='removal_avg')
    plt.legend()
    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell^{TT} [\\mu K^2]$')
    plt.title('average power spectrum')

    plt.figure(2)
    plt.plot(ell_arr, np.sqrt(pcn_var), label='pcn_std')
    plt.plot(ell_arr, np.sqrt(cn_var), label='cn_std')
    plt.plot(ell_arr, np.sqrt(inpaint_var), label='inpaint_std')
    plt.plot(ell_arr, np.sqrt(removal_var), label='removal_std')
    plt.legend()
    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell^{TT} [\\mu K^2]$')
    plt.title('standard deviation power spectrum')

    plt.figure(3)
    plt.plot(ell_arr, delta_ps, label='delta_ps')
    plt.plot(ell_arr, delta_removal, label='delta_removal')
    plt.plot(ell_arr, delta_inpaint, label='delta_inpaint')
    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell^{TT} [\\mu K^2]$')
    plt.title('difference power spectrum')

    plt.legend()
    plt.show()

main()
# plot_dl()



