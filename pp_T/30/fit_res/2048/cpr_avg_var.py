import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import pymaster as nmt
from pathlib import Path

lmax = 300
l = np.arange(lmax+1)
nside = 2048
df = pd.read_csv('../../../../FGSim/FreqBand')

freq = df.at[0, 'freq']
beam = df.at[0, 'beam']
print(f'{freq=}, {beam=}')
bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,0]


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

def gen_ps_remove_map_pcfn(rlz_idx, mask, m_cmb_fg_noise):

    m_res = np.load(f'./ps_cmb_noise_residual/2sigma/map{rlz_idx}.npy')
    m_all = m_res + m_cmb_fg_noise
    
    # hp.orthview(m_all * mask, rot=[100,50,0], half_sky=True)
    # plt.show()
    return m_all * mask

def gen_inpaint_res_map_pcn(rlz_idx, mask, m_cmb_noise):

    m_inpaint = hp.read_map(f'./INPAINT/output/pcn/2sigma/{rlz_idx}.fits', field=0)
    m_res = m_inpaint - m_cmb_noise
    
    # hp.orthview(m_res * mask, rot=[100,50,0], half_sky=True)
    # plt.show()
    return m_res * mask


def main_pcn():
    bin_mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
    apo_mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')

    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax, fold=0.2)
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

        # plt.plot(ell_arr, dl_c, label='cmb', marker='o')
        # plt.plot(ell_arr, dl_pcn, label='ps cmb noise', marker='o')
        # plt.plot(ell_arr, dl_cn, label='cmb noise', marker='o')
        # plt.plot(ell_arr, dl_inpaint, label='inpaint', marker='o')
        # plt.plot(ell_arr, dl_removal, label='removal', marker='o')
        # plt.xlabel('$\\ell$')
        # plt.ylabel('$\\Delta D_\\ell^{TT} [\\mu K^2]$')
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

    path_avg_var = Path(f'./avg_var')
    path_avg_var.mkdir(exist_ok=True, parents=True)

    np.save(path_avg_var / Path(f'inpaint_avg.npy'), dl_inpaint_avg)
    np.save(path_avg_var / Path(f'inpaint_var.npy'), dl_inpaint_var)

    np.save(path_avg_var / Path(f'removal_avg.npy'), dl_removal_avg)
    np.save(path_avg_var / Path(f'removal_var.npy'), dl_removal_var)

    np.save(path_avg_var / Path(f'pcn_avg.npy'), dl_pcn_avg)
    np.save(path_avg_var / Path(f'pcn_var.npy'), dl_pcn_var)

    np.save(path_avg_var / Path(f'cn_avg.npy'), dl_cn_avg)
    np.save(path_avg_var / Path(f'cn_var.npy'), dl_cn_var)


def main_pcfn():
    bin_mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
    apo_mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')

    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax, fold=0.2)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    dl_inpaint_list = []
    dl_removal_list = []
    dl_pcfn_list = []
    dl_cfn_list = []
    dl_c_list = []
    dl_cf_list = []

    m_f = np.load(f'../../../../fitdata/2048/FG/{freq}/fg.npy')[0].copy()
    dl_f = calc_dl_from_scalar_map(m_f, bl, apo_mask, bin_dl, masked_on_input=False)
    for rlz_idx in range(100):
        print(f'{rlz_idx=}')

        m_cfn = np.load(f'../../../../fitdata/synthesis_data/2048/CMBFGNOISE/{freq}/{rlz_idx}.npy')[0].copy()
        # m_pcfn = np.load(f'../../../../fitdata/synthesis_data/2048/PSCMBFGNOISE/{freq}/{rlz_idx}.npy')[0].copy()
        m_c = np.load(f'../../../../fitdata/2048/CMB/{freq}/{rlz_idx}.npy')[0].copy()
        m_cf = np.load(f'../../../../fitdata/synthesis_data/2048/CMBFG/{freq}/{rlz_idx}.npy')[0].copy()
        # m_inpaint = hp.read_map(f'./INPAINT/output/pcfn/2sigma/{rlz_idx}.fits', field=0) * bin_mask * apo_mask
        # m_removal = gen_ps_remove_map_pcfn(rlz_idx=rlz_idx, mask=bin_mask, m_cmb_fg_noise=m_cfn) * apo_mask

        # dl_c = calc_dl_from_scalar_map(m_c, bl, apo_mask, bin_dl, masked_on_input=False)
        # dl_cfn = calc_dl_from_scalar_map(m_cfn, bl, apo_mask, bin_dl, masked_on_input=False)
        dl_cf = calc_dl_from_scalar_map(m_cf, bl, apo_mask, bin_dl, masked_on_input=False)
        # dl_pcfn = calc_dl_from_scalar_map(m_pcfn, bl, apo_mask, bin_dl, masked_on_input=False)
        # dl_inpaint = calc_dl_from_scalar_map(m_inpaint, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=True)
        # dl_removal = calc_dl_from_scalar_map(m_removal, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=True)

        # plt.plot(ell_arr, dl_c, label='cmb', marker='o')
        # plt.plot(ell_arr, dl_pcfn, label='ps cmb fg noise', marker='o')
        # plt.plot(ell_arr, dl_cfn, label='cmb fg noise', marker='o')
        # plt.plot(ell_arr, dl_inpaint, label='inpaint', marker='o')
        # plt.plot(ell_arr, dl_removal, label='removal', marker='o')
        # plt.plot(ell_arr, dl_f, label='diffuse fg', marker='o')
        # plt.xlabel('$\\ell$')
        # plt.ylabel('$\\Delta D_\\ell^{TT} [\\mu K^2]$')
        # plt.ylabel('$D_\\ell^{TT} [\\mu K^2]$')
        # plt.legend()
        # # plt.savefig('./fig/pcn/cpr_ps.png', dpi=300)
        # plt.show()

        # dl_inpaint_list.append(dl_inpaint)
        # dl_removal_list.append(dl_removal)
        # dl_pcfn_list.append(dl_pcfn)
        # dl_cfn_list.append(dl_cfn)
        # dl_c_list.append(dl_c)
        dl_cf_list.append(dl_cf)

    # dl_inpaint_arr = np.asarray(dl_inpaint_list)
    # dl_removal_arr = np.asarray(dl_removal_list)
    # dl_pcfn_arr = np.asarray(dl_pcfn_list)
    # dl_cfn_arr = np.asarray(dl_cfn_list)
    # dl_c_arr = np.asarray(dl_c_list)
    dl_cf_arr = np.asarray(dl_cf_list)

    # dl_inpaint_avg = np.mean(dl_inpaint_arr, axis=0)
    # dl_inpaint_var = np.var(dl_inpaint_arr, axis=0)

    # dl_removal_avg = np.mean(dl_removal_arr, axis=0)
    # dl_removal_var = np.var(dl_removal_arr, axis=0)

    # dl_pcfn_avg = np.mean(dl_pcfn_arr, axis=0)
    # dl_pcfn_var = np.var(dl_pcfn_arr, axis=0)

    # dl_cfn_avg = np.mean(dl_cfn_arr, axis=0)
    # dl_cfn_var = np.var(dl_cfn_arr, axis=0)

    # dl_c_avg = np.mean(dl_c_arr, axis=0)
    # dl_c_var = np.var(dl_c_arr, axis=0)

    dl_cf_avg = np.mean(dl_cf_arr, axis=0)
    dl_cf_var = np.var(dl_cf_arr, axis=0)

    path_avg_var = Path(f'./avg_var')
    path_avg_var.mkdir(exist_ok=True, parents=True)

    # np.save(path_avg_var / Path(f'pcfn_inpaint_avg.npy'), dl_inpaint_avg)
    # np.save(path_avg_var / Path(f'pcfn_inpaint_var.npy'), dl_inpaint_var)

    # np.save(path_avg_var / Path(f'pcfn_removal_avg.npy'), dl_removal_avg)
    # np.save(path_avg_var / Path(f'pcfn_removal_var.npy'), dl_removal_var)

    # np.save(path_avg_var / Path(f'pcfn_avg.npy'), dl_pcfn_avg)
    # np.save(path_avg_var / Path(f'pcfn_var.npy'), dl_pcfn_var)

    # np.save(path_avg_var / Path(f'cfn_avg.npy'), dl_cfn_avg)
    # np.save(path_avg_var / Path(f'cfn_var.npy'), dl_cfn_var)

    # np.save(path_avg_var / Path(f'c_avg.npy'), dl_c_avg)
    # np.save(path_avg_var / Path(f'c_var.npy'), dl_c_var)

    np.save(path_avg_var / Path(f'cf_avg.npy'), dl_cf_avg)
    np.save(path_avg_var / Path(f'cf_var.npy'), dl_cf_var)

    # np.save(path_avg_var / Path(f'diffuse_fg.npy'), dl_f)

def main_noise_bias():
    bin_mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
    apo_mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')

    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax, fold=0.2)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    dl_n_list = []

    nstd = np.load(f'../../../../FGSim/NSTDNORTH/2048/{freq}.npy')[0].copy()
    npix_nstd = np.size(nstd)
    for rlz_idx in range(1000):
        print(f'{rlz_idx=}')

        m_noise = nstd * np.random.normal(0, 1, size=(npix_nstd))
        dl_n = calc_dl_from_scalar_map(m_noise, bl, apo_mask, bin_dl, masked_on_input=False)
        dl_n_list.append(dl_n)

    dl_n_arr = np.asarray(dl_n_list)

    dl_n_avg = np.mean(dl_n_arr, axis=0)
    dl_n_var = np.var(dl_n_arr, axis=0)

    path_avg_var = Path(f'./avg_var')
    path_avg_var.mkdir(exist_ok=True, parents=True)

    np.save(path_avg_var / Path(f'n_avg.npy'), dl_n_avg)
    np.save(path_avg_var / Path(f'n_var.npy'), dl_n_var)

def plot_dl_pcn():
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax, fold=0.2)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    bin_mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
    apo_mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')

    pcn_avg = np.load('./pcn_avg_var/pcn_avg.npy')
    pcn_var = np.load('./pcn_avg_var/pcn_var.npy')

    # cn_avg = np.load(f'./pcn_avg_var/cn_avg.npy')
    # cn_var = np.load(f'./pcn_avg_var/cn_var.npy')

    c_avg = np.load(f'./avg_var/c_avg.npy')
    c_var = np.load(f'./avg_var/c_var.npy')

    inpaint_avg = np.load(f'./pcn_avg_var/inpaint_avg.npy')
    inpaint_var = np.load(f'./pcn_avg_var/inpaint_var.npy')

    removal_avg = np.load(f'./pcn_avg_var/removal_avg.npy')
    removal_var = np.load(f'./pcn_avg_var/removal_var.npy')

    n_bias_avg = np.load(f'./avg_var/n_avg.npy')

    denoise_ps = pcn_avg - n_bias_avg
    denoise_inpaint = inpaint_avg - n_bias_avg
    denoise_removal = removal_avg - n_bias_avg

    delta_ps = denoise_ps - c_avg
    delta_inpaint = denoise_inpaint - c_avg
    delta_removal = denoise_removal - c_avg

    ratio_ps = delta_ps / c_avg
    ratio_inpaint = delta_inpaint / c_avg
    ratio_removal = delta_removal / c_avg

    plt.figure(1)
    plt.plot(ell_arr, denoise_ps, label='denoise pcn_avg')
    plt.plot(ell_arr, c_avg, label='c_avg')
    plt.plot(ell_arr, denoise_inpaint, label='denoise inpaint_avg')
    plt.plot(ell_arr, denoise_removal, label='denoise removal_avg')
    plt.legend()
    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell^{TT} [\\mu K^2]$')
    plt.title('average power spectrum')

    plt.figure(2)
    plt.plot(ell_arr, np.sqrt(pcn_var), label='pcn_std')
    plt.plot(ell_arr, np.sqrt(c_var), label='c_std')
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
    plt.ylabel('$\Delta D_\\ell^{TT} [\\mu K^2]$')
    plt.title('difference power spectrum')
    plt.legend()

    plt.figure(4)
    plt.plot(ell_arr, np.abs(ratio_ps), label='ratio_ps')
    plt.plot(ell_arr, np.abs(ratio_removal), label='ratio_removal')
    plt.plot(ell_arr, np.abs(ratio_inpaint), label='ratio_inpaint')
    plt.plot(ell_arr, np.abs(np.sqrt(c_var)/np.abs(c_avg)), label='ratio_cv')
    plt.semilogy()
    plt.xlabel('$\\ell$')
    plt.ylabel('$\Delta D_\\ell^{TT} /D_\\ell^{TT}$')
    plt.title('power spectrum ratio')
    plt.legend()

    plt.show()

def plot_dl_pcfn():
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax, fold=0.2)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    bin_mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
    apo_mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')

    pcfn_avg = np.load('./avg_var/pcfn_avg.npy')
    pcfn_var = np.load('./avg_var/pcfn_var.npy')

    n_bias_avg = np.load('./avg_var/n_avg.npy')
    n_bias_var = np.load('./avg_var/n_var.npy')

    cf_avg = np.load('./avg_var/cf_avg.npy')
    cf_var = np.load('./avg_var/cf_var.npy')

    c_avg = np.load(f'./avg_var/c_avg.npy')
    c_var = np.load(f'./avg_var/c_var.npy')

    diffuse_fg = np.load(f'./avg_var/diffuse_fg.npy')

    inpaint_avg = np.load(f'./avg_var/pcfn_inpaint_avg.npy')
    inpaint_var = np.load(f'./avg_var/pcfn_inpaint_var.npy')

    removal_avg = np.load(f'./avg_var/pcfn_removal_avg.npy')
    removal_var = np.load(f'./avg_var/pcfn_removal_var.npy')

    
    denoise_ps = pcfn_avg - n_bias_avg
    denoise_inpaint = inpaint_avg - n_bias_avg
    denoise_removal = removal_avg - n_bias_avg

    delta_ps = denoise_ps - cf_avg
    delta_inpaint = denoise_inpaint - cf_avg
    delta_removal = denoise_removal - cf_avg

    ratio_ps = delta_ps / cf_avg
    ratio_inpaint = delta_inpaint / cf_avg
    ratio_removal = delta_removal / cf_avg

    plt.figure(1)
    plt.plot(ell_arr, denoise_ps, label='denoise pcfn_avg')
    plt.plot(ell_arr, cf_avg, label='cf_avg')
    plt.plot(ell_arr, c_avg, label='c_avg')
    plt.plot(ell_arr, denoise_inpaint, label='denoise inpaint_avg')
    plt.plot(ell_arr, denoise_removal, label='denoise removal_avg')
    plt.legend()
    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell^{TT} [\\mu K^2]$')
    plt.title('average power spectrum')

    plt.figure(2)
    plt.plot(ell_arr, np.sqrt(pcfn_var), label='pcfn_std')
    plt.plot(ell_arr, np.sqrt(cf_var), label='cf_std')
    plt.plot(ell_arr, np.sqrt(c_var), label='c_std')
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
    plt.ylabel('$\Delta D_\\ell^{TT} [\\mu K^2]$')
    plt.title('difference power spectrum')
    plt.legend()

    plt.figure(4)
    plt.plot(ell_arr, np.abs(ratio_ps), label='ratio_ps')
    plt.plot(ell_arr, np.abs(ratio_removal), label='ratio_removal')
    plt.plot(ell_arr, np.abs(ratio_inpaint), label='ratio_inpaint')
    plt.plot(ell_arr, np.abs(np.sqrt(c_var)/np.abs(c_avg)), label='ratio_cv')
    plt.semilogy()
    plt.xlabel('$\\ell$')
    plt.ylabel('$\Delta D_\\ell^{TT} /D_\\ell^{TT}$')
    plt.title('power spectrum ratio')
    plt.legend()

    plt.show()


# main_pcn()
# main_pcfn()
# main_noise_bias()
# plot_dl_pcn()
plot_dl_pcfn()



