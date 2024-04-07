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
        m = np.load(f'./output/{rlz_idx}.npy')
        pcn = hp.read_map(f'../input/{rlz_idx}.fits', field=0)
        cmb_noise = np.load(f'../../../../../fitdata/synthesis_data/2048/CMBNOISE/155/{rlz_idx}.npy')[0]
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

    # pcn_rlz = hp.read_map('../input/0.fits')
    # mask = hp.read_map('../mask/mask.fits')
    # inp_rlz = np.load('./output_2000/0.npy')

    # dl_pcn_rlz = l*(l+1)*hp.anafast(pcn_rlz, lmax=lmax) / (2*np.pi)
    # dl_mask_rlz = l*(l+1)*hp.anafast(pcn_rlz * mask, lmax=lmax) / (2*np.pi)
    # dl_inp_rlz = l*(l+1)*hp.anafast(inp_rlz, lmax=lmax) / (2*np.pi)



    cn_mean = np.load('./avg_var/cn_mean.npy')
    inp_mean = np.load('./avg_var/inp_mean.npy')
    # inp_mean_1 = np.load('./avg_var_2000/inp_mean.npy')
    # inp_mean_2 = np.load('./avg_var/inp_mean.npy')
    pcn_mean = np.load('./avg_var/pcn_mean.npy')

    cn_var = np.load('./avg_var/cn_var.npy')
    inp_var = np.load('./avg_var/inp_var.npy')
    pcn_var = np.load('./avg_var/pcn_var.npy')

    inp_mean_isap = np.load('../../test_bias/avg_var/inp_mean.npy')

    inp_res = inp_mean - cn_mean
    # inp_res_1 = inp_mean_1 - cn_mean
    # inp_res_2 = inp_mean_2 - cn_mean
    inp_res_isap = inp_mean_isap - cn_mean
    # ps_res = pcn_mean - cn_mean

    # plt.plot(dl_pcn_rlz, label='pcn')
    # plt.plot(dl_mask_rlz, label='pcn mask')
    # plt.plot(dl_inp_rlz, label='inp')

    # plt.plot(cn_mean/bl**2, label='cn')
    # plt.plot(pcn_mean/bl**2, label='pcn')
    # plt.plot(inp_mean/bl**2, label='inp')

    plt.plot(inp_res/bl**2, label='diffuse inp res')
    # plt.plot(inp_res_1/bl**2, label='diffuse 2000 inp res',linestyle=':')
    # plt.plot(inp_res_2/bl**2, label='diffuse 1000 inp res')
    plt.plot(inp_res_isap/bl**2, label='isap inp res')
    plt.plot(np.sqrt(cn_var)/bl**2, label='cv')

    # plt.plot(ps_res/bl**2, label='ps res')


    plt.semilogy()
    plt.xlim(2,1000)
    plt.ylim(-30,300)
    # plt.ylim(-50,7000)
    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell^{TT}$')
    plt.legend()
    plt.show()

def see_map():
    nside=2048
    for i in range(100):
        ipix = np.load('../../source_indices.npy')[i]
        lon, lat = hp.pix2ang(nside=nside,ipix=ipix, lonlat=True)
        m = np.load('./output/0.npy')
        hp.gnomview(m, rot=[lon,lat,0])
        plt.show()


# calc_avg_var()
see_avg_var()
# see_map()








