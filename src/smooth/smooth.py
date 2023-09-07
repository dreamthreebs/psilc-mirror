import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pandas as pd
from pathlib import Path
import glob
from itertools import zip_longest

df = pd.read_csv('../../FGSim/FreqBand')
n_freq = len(df)
print(f'{n_freq = }')
lmax=300
base_beam = 9 # arcmin
nside=512

cmb_all = glob.glob('../../FGSim/CMB/*.npy')
sorted_cmb = sorted(cmb_all, key=lambda x: int(Path(x).stem))
print(f'{sorted_cmb}')

fgnps_all = glob.glob('../../FGSim/FG_noPS/*.npy')
sorted_fgnps = sorted(fgnps_all, key=lambda x: int(Path(x).stem))
print(f'{sorted_fgnps}')


def check_maps_IQU(maps_pos_list:list, lmax:int, nside:int)->None:
    for file in maps_pos_list:
        print(f'{file}')
        IQU = np.load(file)

        TEBalm = hp.map2alm(IQU, lmax=lmax)
        TEB = hp.alm2map(TEBalm, nside=nside)

        for index, (IQUmap_type,TEBmap_type) in enumerate(zip("IQU","TEB")):
            hp.mollview(IQU[index], norm='hist', title=f'{IQUmap_type}');plt.show()
            hp.mollview(TEB[index], norm='hist', title=f'{TEBmap_type}');plt.show()

        cl = hp.anafast(IQU)
        print(f'{cl.shape}')
        l = np.arange(len(cl[0]))
        for index, cl_type in enumerate(['TT','EE','BB']):
            plt.loglog(l*(l+1)*cl[index]/(2*np.pi));plt.title(f'{cl_type}');plt.show()

def check_maps_TEB(maps_pos_list:list, lmax:int, nside:int)->None:
    for file in maps_pos_list:
        print(f'{file}')
        TEB = np.load(file)

        for index, TEBmap_type in enumerate("TEB"):
            hp.mollview(TEB[index], norm='hist', title=f'{TEBmap_type}');plt.show()
            cl = hp.anafast(TEB[index], lmax=lmax)
            l = np.arange(len(cl))
            plt.loglog(l*(l+1)*cl/(2*np.pi));plt.title(f'{TEBmap_type}{TEBmap_type}');plt.show()

def bl_creater(lmax):
    bl_temp_list = []
    bl_grad_list = []
    bl_curl_list = []
    for i in range(n_freq):
        freq = df.at[i,'freq']
        beam = df.at[i,'beam'] # arcmin
        print(f'{freq=},{beam=}')
        bl_temp = hp.gauss_beam(fwhm=np.deg2rad(beam/60), lmax=lmax, pol=True)[:,0]
        bl_grad = hp.gauss_beam(fwhm=np.deg2rad(beam/60), lmax=lmax, pol=True)[:,1]
        bl_curl = hp.gauss_beam(fwhm=np.deg2rad(beam/60), lmax=lmax, pol=True)[:,2]
        # plt.plot(bl);plt.show()
        bl_temp_list.append(bl_temp)
        bl_grad_list.append(bl_grad)
        bl_curl_list.append(bl_curl)
    bl_temp_arr = np.array(bl_temp_list)
    bl_grad_arr = np.array(bl_grad_list)
    bl_curl_arr = np.array(bl_curl_list)
    print(f'{bl_temp_arr.shape}')

    # you can appoint different beam profiles here, the default is gaussian beam
    return bl_temp_arr, bl_grad_arr, bl_curl_arr

bl_std_temp = hp.gauss_beam(fwhm=np.deg2rad(base_beam/60), lmax=lmax, pol=True)[:,0]
bl_std_grad = hp.gauss_beam(fwhm=np.deg2rad(base_beam/60), lmax=lmax, pol=True)[:,1]
bl_std_curl = hp.gauss_beam(fwhm=np.deg2rad(base_beam/60), lmax=lmax, pol=True)[:,2]

def smooth_maps(bl_temp_arr, bl_grad_arr, bl_curl_arr, bl_std_temp, bl_std_grad, bl_std_curl, m_cmb_files=None, m_fg_files=None, m_nstd_files=None, mask=None)->None:
    m_cmb_files = [] if m_cmb_files is None else m_cmb_files
    m_fg_files = [] if m_fg_files is None else m_fg_files
    m_nstd_files = [] if m_nstd_files is None else m_nstd_files

    for index, (m_cmb, m_fg, m_nstd) in enumerate(zip_longest(m_cmb_files, m_fg_files, m_nstd_files, fillvalue=None)):

        cmb = np.load(m_cmb) if m_cmb is not None else np.zeros((3,12*nside**2))
        fg = np.load(m_fg) if m_fg is not None else np.zeros((3,12*nside**2))
        nstd = np.load(m_nstd) if m_nstd is not None else np.zeros((3,12*nside**2))
        noise = nstd * np.random.normal(0,1,(3,12*nside**2)) # TODO

        m = (cmb + fg + noise)

        if mask is not None:
            m = m * mask

        alm_ori = hp.map2alm(m, lmax=lmax)
        alm_temp = hp.almxfl(alm_ori[0], bl_std_temp/bl_temp_arr[index])
        alm_grad = hp.almxfl(alm_ori[1], bl_std_grad/bl_grad_arr[index])
        alm_curl = hp.almxfl(alm_ori[2], bl_std_curl/bl_curl_arr[index])

        smT = hp.alm2map(alm_temp, nside=nside)
        smE = hp.alm2map(alm_grad, nside=nside)
        smB = hp.alm2map(alm_curl, nside=nside)

        smTEB = np.vstack((smT, smE, smB))

        freq = df.at[index, 'freq']

        np.save(f'./SM_FG_noPS/{freq}.npy', smTEB)


if __name__=="__main__":
    # check_maps_IQU(sorted_fgnps, lmax, nside)
    # bl_temp_arr, bl_grad_arr, bl_curl_arr = bl_creater(lmax)
    # smooth_maps(bl_temp_arr, bl_grad_arr, bl_curl_arr, bl_std_temp, bl_std_grad, bl_std_curl, m_cmb_files=None, m_fg_files=sorted_fgnps)

    # sim_all = glob.glob('./SM_SIM/*.npy')
    # sorted_sim = sorted(sim_all, key=lambda x: int(Path(x).stem))
    # print(f'{sorted_sim}')
    
    # check_maps_TEB(sorted_sim, lmax, nside)

    # smooth_maps(bl_temp_arr, bl_grad_arr, bl_curl_arr, bl_std_temp, bl_std_grad, bl_std_curl, m_cmb_files=sorted_cmb, m_fg_files=None)
    smcmb_all = glob.glob('./SM_CMB/*.npy')
    sorted_smcmb = sorted(smcmb_all, key=lambda x: int(Path(x).stem))
    print(f'{sorted_smcmb}')
    check_maps_TEB(sorted_smcmb, lmax, nside)

    # smooth_maps(bl_temp_arr, bl_grad_arr, bl_curl_arr, bl_std_temp, bl_std_grad, bl_std_curl, m_cmb_files=None, m_fg_files=sorted_fgnps)
    # smfgnps_all = glob.glob('./SM_FG_noPS/*.npy')
    # sorted_smfgnps = sorted(smfgnps_all, key=lambda x: int(Path(x).stem))
    # print(f'{sorted_smfgnps}')
    # check_maps_TEB(sorted_smfgnps, lmax, nside)
