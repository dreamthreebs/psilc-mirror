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

def cpr_spectrum_pcn(bin_mask, apo_mask, bl):
    # bin_dl = nmt.NmtBin.from_edges([20,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,1400,1450],[50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,1400,1450,1500], is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=2000,nlb=50, is_Dell=True)
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=2000, fold=0.2)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    rlz_idx = 0

    m_cn = np.load(f'../../../../fitdata/synthesis_data/2048/CMBNOISE/{freq}/{rlz_idx}.npy')[0].copy()
    m_pcn = np.load(f'../../../../fitdata/synthesis_data/2048/PSCMBNOISE/{freq}/{rlz_idx}.npy')[0].copy()
    m_mask = np.load(f'./INPAINT/mask/pcn/2sigma/apodize_mask/0_2.npy')
    # m_mask1 = np.load(f'./INPAINT/mask/pcn/2sigma/apodize_mask/0.npy')
    m_inpaint = hp.read_map(f'./INPAINT/output/pcn/2sigma/{rlz_idx}.fits', field=0) * bin_mask * apo_mask
    # m_removal = gen_ps_remove_map_pcn(rlz_idx=rlz_idx, mask=bin_mask, m_cmb_noise=m_cn) * apo_mask

    dl_cn = calc_dl_from_scalar_map(m_cn, bl, apo_mask, bin_dl, masked_on_input=False)
    # dl_pcn = calc_dl_from_scalar_map(m_pcn, bl, apo_mask, bin_dl, masked_on_input=False)
    # dl_inpaint = calc_dl_from_scalar_map(m_inpaint, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=True)
    # dl_removal = calc_dl_from_scalar_map(m_removal, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=True)
    dl_apo_ps = calc_dl_from_scalar_map(m_cn, bl, apo_mask=m_mask, bin_dl=bin_dl, masked_on_input=False)
    # dl_apo_ps1 = calc_dl_from_scalar_map(m_pcn, bl, apo_mask=m_mask1, bin_dl=bin_dl, masked_on_input=False)

    # plt.plot(ell_arr, dl_pcn, label='ps cmb noise', marker='o')
    plt.plot(ell_arr, dl_cn, label='cmb noise', marker='o')
    # plt.plot(ell_arr, dl_inpaint, label='inpaint', marker='o')
    # plt.plot(ell_arr, dl_removal, label='removal', marker='o')
    plt.plot(ell_arr, dl_apo_ps, label='apo mask ps', marker='o')
    # plt.plot(ell_arr, dl_apo_ps1, label='apo mask ps 1', marker='o')


    plt.xlabel('$\\ell$')
    # plt.ylabel('$\\Delta D_\\ell^{TT} [\\mu K^2]$')
    plt.ylabel('$D_\\ell^{TT} [\\mu K^2]$')

    plt.legend()
    # plt.savefig('./fig/pcn/cpr_ps.png', dpi=300)
    plt.show()

def cpr_pseudo_spectrum_pcn(bin_mask, apo_mask, bl):

    lmax = 2000
    l = np.arange(lmax+1)
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)

    m_inpaint = hp.read_map('./INPAINT/output/pcn/2sigma/0.fits', field=0) * bin_mask * apo_mask
    m_removal = gen_ps_remove_map_pcn(rlz_idx=0, mask=bin_mask) * apo_mask

    m_ps_cmb_noise = np.load(f'../../../../fitdata/synthesis_data/2048/PSCMBNOISE/{freq}/0.npy')[0] * apo_mask
    m_cmb_noise = np.load(f'../../../../fitdata/synthesis_data/2048/CMBNOISE/{freq}/0.npy')[0] * apo_mask

    dl_inpaint = l * (l+1) * hp.anafast(m_inpaint, lmax=lmax) / (2 * np.pi) / bl**2
    dl_removal = l * (l+1) * hp.anafast(m_removal, lmax=lmax) / (2 * np.pi) / bl**2
    dl_pcn =  l * (l+1) * hp.anafast(m_ps_cmb_noise, lmax=lmax) / (2 * np.pi) / bl**2
    dl_cn =  l * (l+1) * hp.anafast(m_cmb_noise, lmax=lmax) / (2 * np.pi) / bl**2

    delta_dl_ps = dl_pcn - dl_cn
    delta_dl_inpaint = dl_inpaint - dl_cn
    delta_dl_removal = dl_removal - dl_cn


    plt.title(f'{freq}GHz')
    plt.plot(l, dl_inpaint, label='inpaint')
    plt.plot(l, dl_removal, label='removal')
    plt.plot(l, dl_pcn, label='pcn')
    plt.plot(l, dl_cn, label='cn')

    plt.plot(l, delta_dl_ps, label='$\\Delta$ dl ps')
    plt.plot(l, delta_dl_inpaint, label='$\\Delta$ dl inpaint')
    plt.plot(l, delta_dl_removal, label='$\\Delta$ dl removal')
    # plt.semilogy()

    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell^{TT} [\\mu K^2]$')

    plt.legend()
    # plt.savefig('./fig/pcn/cpr_ps.png', dpi=300)
    plt.show()

def cpr_pseudo_spectrum_pcfn(bin_mask, apo_mask, bl):

    lmax = 2000
    l = np.arange(lmax+1)
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)

    m_ps_cmb_fg_noise = np.load(f'../../../../fitdata/synthesis_data/2048/PSCMBFGNOISE/{freq}/0.npy')[0]
    m_cmb_fg_noise = np.load(f'../../../../fitdata/synthesis_data/2048/CMBFGNOISE/{freq}/0.npy')[0]
    m_cmb_noise = np.load(f'../../../../fitdata/synthesis_data/2048/CMBNOISE/{freq}/0.npy')[0]

    m_inpaint = hp.read_map('./INPAINT/output/pcfn/2sigma/0.fits', field=0) * bin_mask * apo_mask
    m_removal = gen_ps_remove_map_pcfn(rlz_idx=0, mask=bin_mask, m_cmb_fg_noise=m_cmb_fg_noise) * apo_mask

    dl_inpaint = l * (l+1) * hp.anafast(m_inpaint, lmax=lmax) / (2 * np.pi) / bl**2
    dl_removal = l * (l+1) * hp.anafast(m_removal, lmax=lmax) / (2 * np.pi) / bl**2
    dl_pcfn =  l * (l+1) * hp.anafast(m_ps_cmb_fg_noise*apo_mask, lmax=lmax) / (2 * np.pi) / bl**2
    dl_cfn =  l * (l+1) * hp.anafast(m_cmb_fg_noise*apo_mask, lmax=lmax) / (2 * np.pi) / bl**2
    dl_cn =  l * (l+1) * hp.anafast(m_cmb_noise*apo_mask, lmax=lmax) / (2 * np.pi) / bl**2

    delta_dl_ps = dl_pcfn - dl_cfn
    delta_dl_inpaint = dl_inpaint - dl_cfn
    delta_dl_removal = dl_removal - dl_cfn


    plt.title(f'{freq}GHz')
    plt.plot(l, dl_inpaint, label='inpaint')
    plt.plot(l, dl_removal, label='removal')
    plt.plot(l, dl_pcfn, label='pcfn')
    plt.plot(l, dl_cfn, label='cfn')
    plt.plot(l, dl_cn, label='cn')


    plt.plot(l, delta_dl_ps, label='$\\Delta$ dl ps')
    plt.plot(l, delta_dl_inpaint, label='$\\Delta$ dl inpaint')
    plt.plot(l, delta_dl_removal, label='$\\Delta$ dl removal')
    # plt.semilogy()

    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell^{TT} [\\mu K^2]$')

    plt.legend()
    # plt.savefig('./fig/pcn/cpr_ps.png', dpi=300)
    plt.show()




def cpr_spectrum_pcfn(bin_mask, apo_mask, bl):
    bin_dl = nmt.NmtBin.from_edges([20,50,100,150,200,250,300,350,400,450,500,550,600],[50,100,150,200,250,300,350,400,450,500,550,600,650], is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    m_inpaint = hp.read_map('./for_inpainting/output/pcfn/0.fits', field=0) * bin_mask
    m_removal = gen_ps_remove_map_pcfn(rlz_idx=0, mask=bin_mask)

    dl_inpaint = calc_dl_from_scalar_map(m_inpaint, bl, apo_mask, bin_dl)
    dl_removal = calc_dl_from_scalar_map(m_removal, bl, apo_mask, bin_dl)

    m_ps_cmb_fg_noise = np.load('../../../../fitdata/synthesis_data/2048/PSCMBFGNOISE/40/0.npy')[0]
    dl_ps_cmb_fg_noise = calc_dl_from_scalar_map(m_ps_cmb_fg_noise, bl, apo_mask, bin_dl)

    m_cmb_fg_noise = np.load('../../../../fitdata/synthesis_data/2048/CMBFGNOISE/40/0.npy')[0]
    dl_cmb_fg_noise = calc_dl_from_scalar_map(m_cmb_fg_noise, bl, apo_mask, bin_dl)

    m_cmb = np.load('../../../../fitdata/2048/CMB/40/0.npy')[0]
    m_fg = np.load('../../../../fitdata/2048/FG/40/fg.npy')[0]

    dl_cmb_fg = calc_dl_from_scalar_map(m_cmb + m_fg, bl, apo_mask, bin_dl)
    dl_cmb = calc_dl_from_scalar_map(m_cmb, bl, apo_mask, bin_dl)

    dl_fg = calc_dl_from_scalar_map(m_fg, bl, apo_mask, bin_dl)

    m_ps = np.load('../../../../fitdata/2048/PS/40/ps.npy')[0]
    dl_ps = calc_dl_from_scalar_map(m_ps, bl, apo_mask, bin_dl)

    m_ps_res = np.load('./ps_cmb_noise_residual/2sigma/map0.npy')
    dl_ps_res = calc_dl_from_scalar_map(m_ps_res, bl, apo_mask, bin_dl)


    plt.plot(ell_arr, dl_inpaint, label='inpaint', marker='o')
    plt.plot(ell_arr, dl_removal, label='removal', marker='o')
    plt.plot(ell_arr, dl_ps_cmb_fg_noise, label='ps+cmb+fg+noise', marker='o')
    plt.plot(ell_arr, dl_cmb_fg_noise, label='cmb+fg+noise', marker='o')
    plt.plot(ell_arr, dl_cmb, label='cmb', marker='o')
    plt.plot(ell_arr, dl_fg, label='diffuse fg', marker='o')

    plt.plot(ell_arr, dl_ps, label='ps', marker='o')
    plt.plot(ell_arr, dl_ps_res, label='ps removal res', marker='o')

    plt.semilogy()
    plt.xlabel('l')
    plt.ylabel('Dl')

    plt.legend()
    plt.savefig('./fig/pcfn/cpr_ps.pdf')
    plt.show()



if __name__ == '__main__':

    lmax = 2000
    l = np.arange(lmax+1)
    nside = 2048
    df = pd.read_csv('../../../../FGSim/FreqBand')

    freq = df.at[7, 'freq']
    beam = df.at[7, 'beam']
    print(f'{freq=}, {beam=}')
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,0]

    # apo_mask = np.load('../src/mask/north/APOMASK2048C1_8.npy')
    bin_mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
    apo_mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')


    cpr_spectrum_pcn(bin_mask, apo_mask, bl)
    # cpr_pseudo_spectrum_pcfn(bin_mask, apo_mask, bl)
    # cpr_spectrum_pcfn(bin_mask, apo_mask, bl)








