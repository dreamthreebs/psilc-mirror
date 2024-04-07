import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax = 2000
l = np.arange(lmax+1)
bl = hp.gauss_beam(fwhm=np.deg2rad(17)/60, lmax=lmax)

def calc_avg_var():
    inp_list = []
    pcn_list = []
    cn_list = []
    
    for rlz_idx in range(100):
        print(f'{rlz_idx=}')
        m = hp.read_map(f'./output/{rlz_idx}.fits', field=0)
        pcn = hp.read_map(f'./input/{rlz_idx}.fits', field=0)
        cmb_noise = np.load(f'../../../../fitdata/synthesis_data/2048/CMBNOISE/155/{rlz_idx}.npy')[0]
        dl = l*(l+1)*hp.anafast(m, lmax=lmax)/(2*np.pi)
        dl_pcn = l*(l+1)*hp.anafast(pcn, lmax=lmax)/(2*np.pi)
        dl_cn = l*(l+1)*hp.anafast(cmb_noise, lmax=lmax)/(2*np.pi)
    
        plt.plot(dl/bl**2, label='inpaint')
        plt.plot(dl_cn/bl**2, label='cn')
        plt.legend()
        plt.show()

        # hp.mollview(m)
        # plt.show()
    
        inp_list.append(dl)
        pcn_list.append(dl_pcn)
        cn_list.append(dl_cn)
    
    inp_arr = np.array(inp_list)
    pcn_arr = np.array(pcn_list)
    cn_arr = np.array(cn_list)
    
    inp_mean = np.mean(inp_arr, axis=0)
    inp_var = np.var(inp_arr, axis=0)
    
    pcn_mean = np.mean(pcn_arr, axis=0)
    pcn_var = np.var(pcn_arr, axis=0)
    
    cn_mean = np.mean(cn_arr, axis=0)
    cn_var = np.var(cn_arr, axis=0)
    
    np.save('./avg_var/inp_mean.npy', inp_mean)
    np.save('./avg_var/inp_var.npy', inp_var)
    
    np.save('./avg_var/pcn_mean.npy', pcn_mean)
    np.save('./avg_var/pcn_var.npy', pcn_var)
    
    np.save('./avg_var/cn_mean.npy', cn_mean)
    np.save('./avg_var/cn_var.npy', cn_var)

def see_avg_var():

    cn_mean = np.load('./avg_var/cn_mean.npy')
    inp_mean = np.load('./avg_var/inp_mean.npy')
    pcn_mean = np.load('./avg_var/pcn_mean.npy')

    cn_var = np.load('./avg_var/cn_var.npy')
    inp_var = np.load('./avg_var/inp_var.npy')
    pcn_var = np.load('./avg_var/pcn_var.npy')

    inp_res = inp_mean - cn_mean
    ps_res = pcn_mean - cn_mean


    plt.plot(cn_mean/bl**2, label='cn')
    plt.plot(pcn_mean/bl**2, label='pcn')
    plt.plot(inp_mean/bl**2, label='inp')

    # plt.plot(inp_res/bl**2, label='inp res')
    # plt.plot(ps_res/bl**2, label='ps res')

    plt.xlim(2,1000)
    # plt.ylim(-30,30)
    plt.ylim(-50,7000)
    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell^{TT}$')
    plt.legend()
    plt.show()

# calc_avg_var()
see_avg_var()








