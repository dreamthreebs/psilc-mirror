import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt
import os
import ipdb
import pandas as pd

def calc_dl_from_scalar_map(scalar_map, bl, apo_mask, bin_dl):
    scalar_field = nmt.NmtField(apo_mask, [scalar_map], beam=bl, masked_on_input=False)
    dl = nmt.compute_full_master(scalar_field, scalar_field, bin_dl)
    return dl[0]

def check_full_sky_cl(cmb, ps, lmax, bl):
    ''' you can see here the point source powerspectrum is very large due to the milky way point source '''

    cl_cmb = hp.anafast(cmb, lmax=lmax)
    cl_ps = hp.anafast(ps, lmax=lmax)
    
    print(f'{cl_cmb.shape=}')
    print(f'{cl_ps.shape=}')
    
    plt.semilogy(l*(l+1)*cl_cmb[2]/(2*np.pi)/bl**2)
    plt.semilogy(l*(l+1)*cl_ps[2]/(2*np.pi)/bl**2)
    plt.show()

def check_partial_sky_cl(cmb, ps, lmax, bl, nside, apo_mask):

    bin_dl = nmt.NmtBin.from_edges([20,50,100,150,200,250,300,350,400,450,500,550,600],[50,100,150,200,250,300,350,400,450,500,550,600,650], is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    bmap_cmb = hp.alm2map(hp.map2alm(cmb, lmax=lmax)[2], nside=nside)
    cl_cmb = hp.anafast(bmap_cmb, lmax=lmax)
    cl_lens = np.load('./lensedB.npy')
    cl_r001 = np.load('./r001.npy')

    bmap_ps = hp.alm2map(hp.map2alm(ps, lmax=lmax)[2], nside=nside) / 2

    dl_cmb = calc_dl_from_scalar_map(bmap_cmb, bl, apo_mask, bin_dl)
    dl_ps = calc_dl_from_scalar_map(bmap_ps, bl, apo_mask, bin_dl)

    plt.plot(ell_arr, dl_cmb, label='from small area cmb', marker='o')
    plt.plot(ell_arr, dl_ps, label='from small area ps', marker='o')
    plt.semilogy(l*(l+1)*cl_cmb/(2*np.pi)/bl[:lmax+1]**2)
    plt.semilogy(l*(l+1)*cl_lens[0:lmax+1]/(2*np.pi), label='theory lensing only')
    plt.semilogy(l*(l+1)*cl_r001[0:lmax+1]/(2*np.pi)/10, label='theory r=0.001')
    plt.semilogy(l*(l+1)*cl_r001[0:lmax+1]/(2*np.pi), label='theory r=0.01')
    plt.semilogy(l*(l+1)*cl_r001[0:lmax+1]/(2*np.pi)*10, label='theory r=0.1')
    plt.xlim(20,700)
    plt.ylim(1e-6,1e-1)
    plt.xlabel('l')
    plt.ylabel('Dl')
    plt.legend(loc='lower right')
    plt.show()

def check_partial_sky_E_mode(cmb, ps, lmax, bl, nside, apo_mask):

    bin_dl = nmt.NmtBin.from_edges([20,50,100,150,200,250,300,350,400,450,500,550,600],[50,100,150,200,250,300,350,400,450,500,550,600,650], is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=50, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    emap_cmb = hp.alm2map(hp.map2alm(cmb, lmax=lmax)[1], nside=nside)
    cl_cmb = hp.anafast(emap_cmb, lmax=lmax)
    cl_lens = np.load('./lenseE.npy')

    emap_ps = hp.alm2map(hp.map2alm(ps, lmax=lmax)[1], nside=nside)

    dl_cmb = calc_dl_from_scalar_map(emap_cmb, bl, apo_mask, bin_dl)
    dl_ps = calc_dl_from_scalar_map(emap_ps, bl, apo_mask, bin_dl)

    plt.plot(ell_arr, dl_cmb, label='from small area cmb', marker='o')
    plt.plot(ell_arr, dl_ps, label='from small area ps', marker='o')
    plt.semilogy(l*(l+1)*cl_cmb/(2*np.pi)/bl[:lmax+1]**2, label='one realization cmb')
    plt.semilogy(l*(l+1)*cl_lens[0:lmax+1]/(2*np.pi), label='theory lensed')
    plt.xlim(20,700)
    plt.ylim(1e-6,1e2)
    plt.xlabel('l')
    plt.ylabel('Dl')
    plt.legend(loc='lower right')
    plt.show()


def gen_tensor_r(r_val):
    import camb

    camb_path = os.path.dirname(camb.__file__)
    print(f'{camb_path}')
    print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))
    pars = camb.read_ini(os.path.join(camb_path,'inifiles','tensor.ini'))
    pars.set_for_lmax(2000, lens_potential_accuracy=2);
    print(f'{pars=}')
    # ipdb.set_trace()
    # pars.InitPower.r = r_val
    results = camb.get_results(pars)
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
    totcl = powers['total']
    unlensedcl = powers['unlensed_total']
    l = np.arange(totcl.shape[0])

    # for index, flag in enumerate(['TT', 'EE', 'BB']):
    #     plt.loglog(l*(l*1)*totcl[:,index]/(2*np.pi),label=f'{flag} lensed')
    #     plt.loglog(l*(l*1)*unlensedcl[:,index]/(2*np.pi),label=f'{flag} unlensed')

    # plt.loglog(l*(l*1)*unlensedcl[:,2]/(2*np.pi),label=f'r={r_val}')
    plt.loglog(l*(l*1)*totcl[:,2]/(2*np.pi),label=f'r=0')

    np.save('lensing.npy', totcl[:,2])

    plt.legend()
    plt.xlabel('l')
    plt.ylabel('Dl')
    
    plt.show()

def gen_E_mode():
    import camb

    camb_path = os.path.dirname(camb.__file__)
    print(f'{camb_path}')
    print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))
    pars = camb.read_ini(os.path.join(camb_path,'inifiles','tensor.ini'))
    pars.set_for_lmax(2000, lens_potential_accuracy=2);
    print(f'{pars=}')
    # ipdb.set_trace()
    # pars.InitPower.r = r_val
    results = camb.get_results(pars)
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
    totcl = powers['total']
    unlensedcl = powers['unlensed_total']
    l = np.arange(totcl.shape[0])

    # for index, flag in enumerate(['TT', 'EE', 'BB']):
    #     plt.loglog(l*(l*1)*totcl[:,index]/(2*np.pi),label=f'{flag} lensed')
    #     plt.loglog(l*(l*1)*unlensedcl[:,index]/(2*np.pi),label=f'{flag} unlensed')

    plt.loglog(l*(l*1)*unlensedcl[:,1]/(2*np.pi),label=f'unlensed E')
    plt.loglog(l*(l*1)*totcl[:,1]/(2*np.pi),label=f'lensed E')

    np.save('lenseE.npy', totcl[:,1])
    np.save('unlenseE.npy', unlensedcl[:,1])

    plt.legend()
    plt.xlabel('l')
    plt.ylabel('Dl')
    
    plt.show()



if __name__ == '__main__':

    lmax = 750
    l = np.arange(lmax+1)
    nside = 2048
    df = pd.read_csv('../FGSim/FreqBand')

    freq = df.at[0, 'freq']
    beam = df.at[0, 'beam']
    print(f'{freq=}, {beam=}')
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,2]

    apo_mask = np.load('../src/mask/north/APOMASK2048C1_8.npy')
    cmb = np.load('./2048/CMB/30/0.npy')
    ps = np.load('./2048/PS/30/ps.npy')
    
    print(f'{cmb.shape=}')
    print(f'{ps.shape=}')

    # check_full_sky_cl(cmb, ps, lmax, bl)
    # check_partial_sky_cl(cmb, ps, lmax, bl, nside, apo_mask)
    check_partial_sky_E_mode(cmb, ps, lmax, bl, nside, apo_mask)

    # gen_tensor_r(r_val=0.01)

    # gen_E_mode()




