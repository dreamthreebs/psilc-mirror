import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import pymaster as nmt
from pathlib import Path

def gen_ps_remove_map_pcn(rlz_idx, mask, m_cmb_noise):

    m_res = np.load(f'./ps_cmb_noise_residual/2sigma/map{rlz_idx}.npy')
    m_all = m_res + m_cmb_noise
    
    # hp.orthview(m_all * mask, rot=[100,50,0], half_sky=True)
    # plt.show()
    return m_all * mask

def main():
    lmax = 2000
    l = np.arange(lmax+1)
    nside = 2048
    df = pd.read_csv('../../../../FGSim/FreqBand')

    freq = df.at[6, 'freq']
    beam = df.at[6, 'beam']
    print(f'{freq=}, {beam=}')
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,0]

    bin_mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
    apo_mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')


    dl_inpaint_list = []
    dl_removal_list = []
    dl_pcn_list = []
    dl_cn_list = []
    for rlz_idx in range(100):
        print(f'{rlz_idx=}')
        m_cmb_noise = np.load(f'../../../../fitdata/synthesis_data/2048/CMBNOISE/{freq}/{rlz_idx}.npy')[0].copy()
        m_ps_cmb_noise = np.load(f'../../../../fitdata/synthesis_data/2048/PSCMBNOISE/{freq}/{rlz_idx}.npy')[0].copy()
        m_removal = gen_ps_remove_map_pcn(rlz_idx=rlz_idx, mask=bin_mask, m_cmb_noise=m_cmb_noise) * apo_mask
        m_inpaint = hp.read_map(f'./INPAINT/output/pcn/2sigma/{rlz_idx}.fits', field=0) * bin_mask * apo_mask

        dl_inpaint = l * (l+1) * hp.anafast(m_inpaint, lmax=lmax) / (2 * np.pi) / bl**2
        dl_removal = l * (l+1) * hp.anafast(m_removal, lmax=lmax) / (2 * np.pi) / bl**2
        dl_pcn =  l * (l+1) * hp.anafast(m_ps_cmb_noise*apo_mask, lmax=lmax) / (2 * np.pi) / bl**2
        dl_cn =  l * (l+1) * hp.anafast(m_cmb_noise*apo_mask, lmax=lmax) / (2 * np.pi) / bl**2

        # plt.plot(l, dl_inpaint, label='inpaint')
        # plt.plot(l, dl_removal, label='removal')
        # plt.plot(l, dl_pcn, label='pcn')
        # plt.plot(l, dl_cn, label='cn')
        # plt.legend()
        # plt.show()

        dl_inpaint_list.append(dl_inpaint)
        dl_removal_list.append(dl_removal)
        dl_pcn_list.append(dl_pcn)
        dl_cn_list.append(dl_cn)

    dl_inpaint_arr = np.asarray(dl_inpaint_list)
    dl_removal_arr = np.asarray(dl_removal_list)
    dl_pcn_arr = np.asarray(dl_pcn_list)
    dl_cn_arr = np.asarray(dl_cn_list)

    dl_inpaint_avg = np.sum(dl_inpaint_arr, axis=0)
    dl_inpaint_var = np.var(dl_inpaint_arr, axis=0)

    dl_removal_avg = np.sum(dl_removal_arr, axis=0)
    dl_removal_var = np.var(dl_removal_arr, axis=0)

    dl_pcn_avg = np.sum(dl_pcn_arr, axis=0)
    dl_pcn_var = np.var(dl_pcn_arr, axis=0)

    dl_cn_avg = np.sum(dl_cn_arr, axis=0)
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
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,0]

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
    plt.plot(l, pcn_avg/100, label='pcn_avg')
    plt.plot(l, cn_avg/100, label='cn_avg')
    plt.plot(l, inpaint_avg/100, label='inpaint_avg')
    plt.plot(l, removal_avg/100, label='removal_avg')
    plt.legend()
    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell^{TT} [\\mu K^2]$')
    plt.title('average power spectrum')

    plt.figure(2)
    plt.plot(l, np.sqrt(pcn_var), label='pcn_std')
    plt.plot(l, np.sqrt(cn_var), label='cn_std')
    plt.plot(l, np.sqrt(inpaint_var), label='inpaint_std')
    plt.plot(l, np.sqrt(removal_var), label='removal_std')
    plt.legend()
    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell^{TT} [\\mu K^2]$')
    plt.title('standard deviation power spectrum')

    plt.figure(3)
    plt.plot(l, delta_ps/100, label='delta_ps')
    plt.plot(l, delta_removal/100, label='delta_removal')
    plt.plot(l, delta_inpaint/100, label='delta_inpaint')
    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell^{TT} [\\mu K^2]$')
    plt.title('difference power spectrum')

    plt.legend()
    plt.show()

# main()
plot_dl()



