import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

def check_contribution_from_PS():
    lmax = 1000
    l = np.arange(lmax+1)
    freq = 155
    
    beam1 = 17
    beam2 = 30
    
    bl_1 = hp.gauss_beam(fwhm=np.deg2rad(beam1)/60, lmax=lmax)
    bl_2 = hp.gauss_beam(fwhm=np.deg2rad(beam2)/60, lmax=lmax)
    
    
    radio_ps = hp.read_map(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/{freq}GHz/strongradiops_map_{freq}GHz.fits', field=0)
    ir_ps = hp.read_map(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/{freq}GHz/strongirps_map_{freq}GHz.fits', field=0)
    
    # hp.mollview(radio_ps)
    # plt.show()
    
    m_ps = radio_ps + ir_ps
    
    sm_ps_1 = hp.smoothing(m_ps, fwhm=np.deg2rad(beam1)/60, lmax=lmax)
    sm_ps_2 = hp.smoothing(m_ps, fwhm=np.deg2rad(beam2)/60, lmax=lmax)
    
    cl1 = hp.anafast(sm_ps_1, lmax=lmax)
    cl2 = hp.anafast(sm_ps_2, lmax=lmax)
    
    plt.plot(l*(l+1)*cl1/(2*np.pi)/bl_1**2, label=f'beam = {beam1}')
    plt.plot(l*(l+1)*cl2/(2*np.pi)/bl_2**2, label=f'beam = {beam2}', linestyle='--')
    plt.legend()
    plt.xlabel('l')
    plt.ylabel('DL_TT')
    plt.show()

def check_ps_from_diffrent_freq():
    lmax = 1000
    l = np.arange(lmax+1)
    freq_list = [30,40,85,95,145,155,215,270]

    beam = 17
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)

    cl_cmb = np.load('../../../src/cmbsim/cmbdata/cmbcl.npy')[0:lmax+1, 0]
    plt.semilogy(l*(l+1)*cl_cmb/(2*np.pi), label=f'real_cmb', color='black')
    
    for freq in freq_list:
        radio_ps = hp.read_map(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/{freq}GHz/strongradiops_map_{freq}GHz.fits', field=0)
        ir_ps = hp.read_map(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/{freq}GHz/strongirps_map_{freq}GHz.fits', field=0)
    
        # hp.mollview(radio_ps)
        # plt.show()
    
        m_ps = radio_ps + ir_ps
    
        sm_ps = hp.smoothing(m_ps, fwhm=np.deg2rad(beam)/60, lmax=lmax)

        cl = hp.anafast(sm_ps, lmax=lmax)
    
        plt.semilogy(l*(l+1)*cl/(2*np.pi)/bl**2, label=f'freq={freq}')
def check_fg_from_diffrent_freq():
    lmax = 1000
    l = np.arange(lmax+1)
    freq_list = [30,40,85,95,145,155,215,270]

    beam = 17
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)

    cl_cmb = np.load('../../../src/cmbsim/cmbdata/cmbcl.npy')[0:lmax+1, 0]
    plt.semilogy(l*(l+1)*cl_cmb/(2*np.pi), label=f'real_cmb', color='black')
    
    for freq in freq_list:
        synchrotron = hp.read_map(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/155GHz/synchrotron_map_155GHz.fits', field=0)
        thermal_dust = hp.read_map(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/155GHz/thermaldust_map_155GHz.fits', field=0)
    
        # hp.mollview(radio_ps)
        # plt.show()
    
        m_ps = radio_ps + ir_ps
    
        sm_ps = hp.smoothing(m_ps, fwhm=np.deg2rad(beam)/60, lmax=lmax)

        cl = hp.anafast(sm_ps, lmax=lmax)
    
        plt.semilogy(l*(l+1)*cl/(2*np.pi)/bl**2, label=f'freq={freq}')



    plt.legend()
    plt.xlabel('l')
    plt.ylabel('DL_TT')
    plt.show()



# check_contribution_from_PS()
check_ps_from_diffrent_freq()


