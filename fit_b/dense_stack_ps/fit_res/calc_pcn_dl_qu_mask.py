import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import pymaster as nmt

from pathlib import Path

lmax = 500
l = np.arange(lmax+1)
nside = 2048
rlz_idx=0
threshold = 3

df = pd.read_csv('../../../FGSim/FreqBand')
freq = df.at[0, 'freq']
beam = df.at[0, 'beam']
print(f'{freq=}, {beam=}')

bin_mask = np.load('../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
apo_mask = np.load('../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')
ps_mask = np.load(f'../inpainting/mask/apo_ps_mask.npy')

noise_seeds = np.load('../../seeds_noise_2k.npy')
cmb_seeds = np.load('../../seeds_cmb_2k.npy')

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

def calc_dl_from_pol_map(m_q, m_u, bl, apo_mask, bin_dl, masked_on_input, purify_b):
    pol_field = nmt.NmtField(apo_mask, [m_q, m_u], beam=bl, masked_on_input=masked_on_input, purify_b=purify_b)
    dl = nmt.compute_full_master(pol_field, pol_field, bin_dl)
    return dl[3]

def gen_map(rlz_idx):
    npix = hp.nside2npix(nside=nside)
    ps = np.load('../data/ps_67.npy')

    nstd = np.load('../../../FGSim/NSTDNORTH/2048/30.npy')
    np.random.seed(seed=noise_seeds[rlz_idx])
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3,npix))
    print(f"{np.std(noise[1])=}")

    # cmb_iqu = np.load(f'../../fitdata/2048/CMB/215/{rlz_idx}.npy')
    # cls = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')
    cls = np.load('../../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    np.random.seed(seed=cmb_seeds[rlz_idx])
    # cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=1999)
    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=3*nside-1)

    pcn = noise + ps + cmb_iqu
    cn = noise + cmb_iqu
    c = cmb_iqu
    n = noise
    return pcn, cn, c, n

def cpr_spectrum_pcn_b(bin_mask, apo_mask):

    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax, fold=0.2)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    # m_c = np.load(f'../../../../fitdata/2048/CMB/{freq}/{rlz_idx}.npy')
    # m_cn = np.load(f'../../../../fitdata/synthesis_data/2048/CMBNOISE/{freq}/{rlz_idx}.npy')
    # m_pcn = np.load(f'../../../../fitdata/synthesis_data/2048/PSCMBNOISE/{freq}/{rlz_idx}.npy')

    m_pcn, m_cn, m_c, m_n= gen_map(rlz_idx=rlz_idx)

    m_pcn_q = m_pcn[1].copy()
    m_pcn_u = m_pcn[2].copy()
    m_n_q = m_n[1].copy()
    m_n_u = m_n[2].copy()
    print('begin calc dl...')

    dl_qu = calc_dl_from_pol_map(m_q=m_pcn_q, m_u=m_pcn_u, bl=bl, apo_mask=ps_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    dl_qu_n = calc_dl_from_pol_map(m_q=m_n_q, m_u=m_n_u, bl=bl, apo_mask=ps_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)

    path_dl_qu = Path(f'pcn_dl/ACT/curl_yp_ps')
    path_dl_qu_n = Path(f'pcn_dl/ACT/curl_yp_ps_n')
    path_dl_qu.mkdir(parents=True, exist_ok=True)
    path_dl_qu_n.mkdir(parents=True, exist_ok=True)

    np.save(path_dl_qu / Path(f'{rlz_idx}.npy'), dl_qu)
    np.save(path_dl_qu_n / Path(f'{rlz_idx}.npy'), dl_qu_n)


def cpr_spectrum_pcn_e(bin_mask, apo_mask):

    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,1]
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax, fold=0.2)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    # m_c = np.load(f'../../../../fitdata/2048/CMB/{freq}/{rlz_idx}.npy')
    # m_cn = np.load(f'../../../../fitdata/synthesis_data/2048/CMBNOISE/{freq}/{rlz_idx}.npy')
    # m_pcn = np.load(f'../../../../fitdata/synthesis_data/2048/PSCMBNOISE/{freq}/{rlz_idx}.npy')

    # m_c_e = hp.alm2map(hp.map2alm(m_c, lmax=lmax)[1], nside=nside) * bin_mask
    # m_cn_e = hp.alm2map(hp.map2alm(m_cn, lmax=lmax)[1], nside=nside) * bin_mask
    # m_pcn_e = hp.alm2map(hp.map2alm(m_pcn, lmax=lmax)[1], nside=nside) * bin_mask
    # m_removal_e = np.load(f'./pcn_after_removal/{threshold}sigma/E/map_crp_e{rlz_idx}.npy')

    m_ps_e = hp.read_map(f'./inpaint_pcn/{threshold}sigma/EB/E_input/{rlz_idx}.fits') * bin_mask
    m_inp_eb_e = hp.read_map(f'./inpaint_pcn/{threshold}sigma/EB/E_output/{rlz_idx}.fits') * bin_mask
    # m_inp_qu_e = hp.read_map(f'./inpaint_pcn/{threshold}sigma/QU/E/{rlz_idx}.fits') * bin_mask

    # e_min = -20
    # e_max = 20
    # # ### test: checking map
    # hp.orthview(m_cn_e, rot=[100,50,0], half_sky=True, title=' cn e ', min=e_min, max=e_max)
    # # hp.orthview(m_pcn_e, rot=[100,50,0], half_sky=True, title=' pcn e ', min=e_min, max=e_max)
    # hp.orthview(m_removal_e, rot=[100,50,0], half_sky=True, title=' removal e ', min=e_min, max=e_max)
    # hp.orthview(m_ps_e, rot=[100,50,0], half_sky=True, title='ps e', min=e_min, max=e_max)
    # hp.orthview(m_inp_qu_e, rot=[100,50,0], half_sky=True, title='inp qu e', min=e_min, max=e_max)
    # hp.orthview(m_inp_eb_e, rot=[100,50,0], half_sky=True, title='inp eb e', min=e_min, max=e_max)
    # plt.show()


    # plt.show()

    # dl_c_e = calc_dl_from_scalar_map(m_c_e, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)
    # dl_cn_e = calc_dl_from_scalar_map(m_cn_e, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)
    # dl_pcn_e = calc_dl_from_scalar_map(m_pcn_e, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)
    # dl_removal_e = calc_dl_from_scalar_map(m_removal_e, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)

    dl_ps_e = calc_dl_from_scalar_map(m_ps_e, bl, apo_mask=ps_mask, bin_dl=bin_dl, masked_on_input=False)
    dl_inp_eb_e = calc_dl_from_scalar_map(m_inp_eb_e, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)
    # dl_inp_qu_e = calc_dl_from_scalar_map(m_inp_qu_e, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)


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
    path_dl_ps = Path(f'pcn_dl/E/ps_{threshold}sigma')
    path_dl_ps.mkdir(parents=True, exist_ok=True)
    path_dl_inpaint_eb = Path(f'pcn_dl/E/inpaint_eb_{threshold}sigma')
    path_dl_inpaint_eb.mkdir(parents=True, exist_ok=True)

    # path_dl_inpaint_qu = Path(f'pcn_dl/E/inpaint_qu_{threshold}sigma')
    # path_dl_inpaint_qu.mkdir(parents=True, exist_ok=True)

    # np.save(path_dl_c / Path(f'{rlz_idx}.npy'), dl_c_e)
    # np.save(path_dl_cn / Path(f'{rlz_idx}.npy'), dl_cn_e)
    # np.save(path_dl_pcn / Path(f'{rlz_idx}.npy'), dl_pcn_e)
    # np.save(path_dl_removal / Path(f'{rlz_idx}.npy'), dl_removal_e)

    np.save(path_dl_ps / Path(f'{rlz_idx}.npy'), dl_ps_e)
    np.save(path_dl_inpaint_eb / Path(f'{rlz_idx}.npy'), dl_inp_eb_e)
    # np.save(path_dl_inpaint_qu / Path(f'{rlz_idx}.npy'), dl_inp_qu_e)

    # plt.plot(ell_arr, dl_c_e, label='c e', marker='o')
    # plt.plot(ell_arr, dl_cn_e, label='cn e', marker='o')
    # plt.plot(ell_arr, dl_pcn_e, label='pcn e', marker='o')
    # plt.plot(ell_arr, dl_removal_e, label='removal e', marker='o')
    # plt.semilogy()
    # plt.xlabel('$\\ell$')
    # plt.ylabel('$D_\\ell$')
    # plt.legend()
    # plt.show()

def cpr_spectrum_noise_bias_b(bin_mask=bin_mask, apo_mask=apo_mask):
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax, fold=0.2)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    nstd = np.load(f'../../../../FGSim/NSTDNORTH/2048/{freq}.npy')
    noise = nstd * np.random.normal(loc=0, scale=1, size=(nstd.shape[0], nstd.shape[1]))

    ### test: noise map
    # hp.mollview(noise[0])
    # hp.mollview(noise[1])
    # hp.mollview(noise[2])
    # plt.show()

    noise_b = hp.alm2map(hp.map2alm(noise, lmax=lmax)[2], nside=nside) * bin_mask
    dl_n_b = calc_dl_from_scalar_map(noise_b, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)

    path_dl_n = Path(f'pcn_dl/B/n')
    path_dl_n.mkdir(exist_ok=True, parents=True)
    np.save(path_dl_n / Path(f'{rlz_idx}.npy'), dl_n_b)

def cpr_spectrum_noise_bias_e(bin_mask=bin_mask, apo_mask=apo_mask):
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,1]
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax, fold=0.2)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    nstd = np.load(f'../../../../FGSim/NSTDNORTH/2048/{freq}.npy')
    noise = nstd * np.random.normal(loc=0, scale=1, size=(nstd.shape[0], nstd.shape[1]))

    ### test: noise map
    # hp.mollview(noise[0])
    # hp.mollview(noise[1])
    # hp.mollview(noise[2])
    # plt.show()

    noise_e = hp.alm2map(hp.map2alm(noise, lmax=lmax)[1], nside=nside) * bin_mask
    dl_n_e = calc_dl_from_scalar_map(noise_e, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)

    path_dl_n = Path(f'pcn_dl/E/n')
    path_dl_n.mkdir(exist_ok=True, parents=True)
    np.save(path_dl_n / Path(f'{rlz_idx}.npy'), dl_n_e)

def calc_true_noise_bias_b(bin_mask=bin_mask, apo_mask=apo_mask):
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax, fold=0.2)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    noise = np.load(f'../../../../fitdata/2048/NOISE/{freq}/{rlz_idx}.npy')

    ### test: noise map
    # hp.mollview(noise[0])
    # hp.mollview(noise[1])
    # hp.mollview(noise[2])
    # plt.show()

    noise_b = hp.alm2map(hp.map2alm(noise, lmax=lmax)[2], nside=nside) * bin_mask
    dl_n_b = calc_dl_from_scalar_map(noise_b, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)

    path_dl_n = Path(f'pcn_dl/B/n_true')
    path_dl_n.mkdir(exist_ok=True, parents=True)
    np.save(path_dl_n / Path(f'{rlz_idx}.npy'), dl_n_b)

def calc_true_noise_bias_e(bin_mask=bin_mask, apo_mask=apo_mask):
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,1]
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax, fold=0.2)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    noise = np.load(f'../../../../fitdata/2048/NOISE/{freq}/{rlz_idx}.npy')

    ### test: noise map
    # hp.mollview(noise[0])
    # hp.mollview(noise[1])
    # hp.mollview(noise[2])
    # plt.show()

    noise_e = hp.alm2map(hp.map2alm(noise, lmax=lmax)[1], nside=nside) * bin_mask
    dl_n_e = calc_dl_from_scalar_map(noise_e, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)

    path_dl_n = Path(f'pcn_dl/E/n_true')
    path_dl_n.mkdir(exist_ok=True, parents=True)
    np.save(path_dl_n / Path(f'{rlz_idx}.npy'), dl_n_e)


def test_c_b(bin_mask, apo_mask):
    lmax = 1200

    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax, fold=0.2)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()
    print(f'{ell_arr.shape=}')

    m_c = np.load(f'../../../../fitdata/2048/CMB/{freq}/{rlz_idx}.npy')

    m_c_b = hp.alm2map(hp.map2alm(m_c, lmax=lmax)[2], nside=nside) * bin_mask

    dl_c_b = calc_dl_from_scalar_map(m_c_b, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)

    path_dl_c = Path(f'pcn_dl/B/test_c')
    path_dl_c.mkdir(parents=True, exist_ok=True)

    np.save(path_dl_c / Path(f'{rlz_idx}.npy'), dl_c_b)

    # plt.plot(ell_arr, dl_c_b, label='c b', marker='o')
    # plt.plot(ell_arr, dl_cn_b, label='cn b', marker='o')
    # plt.plot(ell_arr, dl_pcn_b, label='pcn b', marker='o')
    # plt.plot(ell_arr, dl_removal_b, label='removal b', marker='o')
    # plt.semilogy()
    # plt.xlabel('$\\ell$')
    # plt.ylabel('$D_\\ell$')
    # plt.legend()
    # plt.show()


def main():

    cpr_spectrum_pcn_b(bin_mask=bin_mask, apo_mask=apo_mask)
    # cpr_spectrum_noise_bias_b()

    # calc_true_noise_bias_b()

    # test_c_b(bin_mask=bin_mask, apo_mask=apo_mask)

main()






