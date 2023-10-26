import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pandas as pd
from pathlib import Path
import glob
import os
import time
from itertools import zip_longest
import matplotlib.colors as mcolors

df = pd.read_csv('../../FGSim/FreqBand')
n_freq = len(df)
print(f'{n_freq = }')
lmax=500
base_beam = 9 # arcmin
nside=512

cmb_all = glob.glob('../../FGSim/CMB/*.npy')
sorted_cmb = sorted(cmb_all, key=lambda x: int(Path(x).stem))
print(f'{sorted_cmb}')

fgnps_all = glob.glob('../../FGSim/FG_noPS/*.npy')
sorted_fgnps = sorted(fgnps_all, key=lambda x: int(Path(x).stem))
print(f'{sorted_fgnps}')

fgps_all = glob.glob('../../FGSim/FG/*.npy')
sorted_fgps = sorted(fgps_all, key=lambda x: int(Path(x).stem))
print(f'{sorted_fgps}')

nstd_all = glob.glob('../../FGSim/NSTDNORTH/*.npy')
sorted_nstd = sorted(nstd_all, key=lambda x: int(Path(x).stem))
print(f'{sorted_nstd}')


bl_std_temp = hp.gauss_beam(fwhm=np.deg2rad(base_beam/60), lmax=lmax, pol=True)[:,0]
bl_std_grad = hp.gauss_beam(fwhm=np.deg2rad(base_beam/60), lmax=lmax, pol=True)[:,1]
bl_std_curl = hp.gauss_beam(fwhm=np.deg2rad(base_beam/60), lmax=lmax, pol=True)[:,2]

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
            # hp.mollview(TEB[index], norm='hist', title=f'{TEBmap_type}');plt.show()
            # hp.mollview(TEB[index], title=f'{TEBmap_type}');plt.show()
            hp.orthview(hp.pixelfunc.ma(TEB[index],badval=0),half_sky=True, rot=[100,50,0],cmap='viridis', title=f'{TEBmap_type}');plt.show()
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

def smooth_maps(bl_temp_arr, bl_grad_arr, bl_curl_arr, bl_std_temp, bl_std_grad, bl_std_curl, m_cmb_files=None, m_fg_files=None, m_noise_files=None, m_nstd_files=None, bin_mask=None, apo_mask=None, nside=512, num_sim=None, save_path=None)->None:
    nside = nside
    n_pix = hp.nside2npix(nside)
    m_cmb_files = [] if m_cmb_files is None else m_cmb_files
    m_fg_files = [] if m_fg_files is None else m_fg_files
    m_noise_files = [] if m_noise_files is None else m_noise_files
    m_nstd_files = [] if m_nstd_files is None else m_nstd_files

    for index, (m_cmb, m_fg, m_noise, m_nstd) in enumerate(zip_longest(m_cmb_files, m_fg_files, m_noise_files,m_nstd_files, fillvalue=None)):

        cmb = np.load(m_cmb) if m_cmb is not None else np.zeros((3,n_pix))
        fg = np.load(m_fg) if m_fg is not None else np.zeros((3,n_pix))
        if (m_noise and m_nstd) == True:
            print('cannot input both noise and nstd')
        noise = np.load(m_noise) if m_noise is not None else np.zeros((3,n_pix))
        nstd =np.load(m_nstd) if m_nstd is not None else np.zeros((3,n_pix))
        noise = nstd * np.random.normal(0,1,(3,n_pix))

        m = (cmb + fg + noise)

        if apo_mask is not None:
            m = m * hp.smoothing(apo_mask, fwhm=np.deg2rad(1))

        alm_ori = hp.map2alm(m, lmax=lmax)
        alm_temp = hp.almxfl(alm_ori[0], bl_std_temp/bl_temp_arr[index])
        alm_grad = hp.almxfl(alm_ori[1], bl_std_grad/bl_grad_arr[index])
        alm_curl = hp.almxfl(alm_ori[2], bl_std_curl/bl_curl_arr[index])

        if bin_mask is not None:
            smT = hp.alm2map(alm_temp, nside=nside) * bin_mask
            smE = hp.alm2map(alm_grad, nside=nside) * bin_mask
            smB = hp.alm2map(alm_curl, nside=nside) * bin_mask
        else:
            smT = hp.alm2map(alm_temp, nside=nside)
            smE = hp.alm2map(alm_grad, nside=nside)
            smB = hp.alm2map(alm_curl, nside=nside)

        smTEB = np.vstack((smT, smE, smB))
        freq = df.at[index, 'freq']

        # np.save(f'./FULL_SKY/SM_lowNOISE/{freq}.npy', smTEB)
        # if num_sim is not None:
            # np.save(f'{save_path}/{num_sim}/{freq}.npy', smTEB)
        # else:
        np.save(f'{save_path}/{freq}.npy', smTEB)

def load_and_check(prefix):
    file = glob.glob(f'{prefix}/*.npy')
    sorted_file = sorted(file, key=lambda x: int(Path(x).stem))
    print(f'{sorted_file}')
    check_maps_TEB(sorted_file, lmax, nside)

def split_maps(path):
    maps = glob.glob(os.path.join(path, '*.npy'))
    map_list = sorted(maps, key=lambda x: int(Path(x).stem))
    T_list=[];E_list=[];B_list=[]
    for file in map_list:
        freq = int(Path(file).stem)
        print(f'split{freq}...')
        m = np.load(file)
        T_list.append(m[0]);E_list.append(m[1]);B_list.append(m[2])

    T=np.array(T_list);E=np.array(E_list);B=np.array(B_list)
    T_PATH = os.path.join(path,'T')
    E_PATH = os.path.join(path,'E')
    B_PATH = os.path.join(path,'B')
    paths = [(T_PATH, T), (E_PATH, E), (B_PATH, B)]
    
    for p, data in paths:
        if not os.path.exists(p):
            os.makedirs(p)
        np.save(os.path.join(p, 'data.npy'), data)


def data_creater(root_path, cmb_data_list=None, fg_data_list=None, nstd_data_list=None, bin_mask=None, apo_mask=None):
    ''' create all data '''
    print('create sim...')
    SIMPATH = os.path.join(root_path, 'SIM')
    if not os.path.exists(SIMPATH):
        os.makedirs(SIMPATH)
    smooth_maps(bl_temp_arr, bl_grad_arr, bl_curl_arr, bl_std_temp, bl_std_grad, bl_std_curl, m_cmb_files=cmb_data_list, m_fg_files=fg_data_list,m_nstd_files=nstd_data_list, save_path=SIMPATH, bin_mask=bin_mask,apo_mask=apo_mask)
    split_maps(SIMPATH)

    print('create cmb...')
    CMBPATH = os.path.join(root_path, 'CMB')
    if not os.path.exists(CMBPATH):
        os.makedirs(CMBPATH)
    smooth_maps(bl_temp_arr, bl_grad_arr, bl_curl_arr, bl_std_temp, bl_std_grad, bl_std_curl, m_cmb_files=cmb_data_list, m_fg_files=None,m_nstd_files=None, save_path=CMBPATH, bin_mask=bin_mask, apo_mask=apo_mask)
    split_maps(CMBPATH)

    print('create fg...')
    FGPATH = os.path.join(root_path, 'FG')
    if not os.path.exists(FGPATH):
        os.makedirs(FGPATH)
    smooth_maps(bl_temp_arr, bl_grad_arr, bl_curl_arr, bl_std_temp, bl_std_grad, bl_std_curl, m_cmb_files=None, m_fg_files=fg_data_list,m_nstd_files=None, save_path=FGPATH, bin_mask=bin_mask, apo_mask=apo_mask)
    split_maps(FGPATH)

    print('create noise...')
    NOISEPATH = os.path.join(root_path, 'NOISE')
    if not os.path.exists(NOISEPATH):
        os.makedirs(NOISEPATH)
    smooth_maps(bl_temp_arr, bl_grad_arr, bl_curl_arr, bl_std_temp, bl_std_grad, bl_std_curl, m_cmb_files=None, m_fg_files=None,m_nstd_files=nstd_data_list, save_path=NOISEPATH, bin_mask=bin_mask, apo_mask=apo_mask)
    split_maps(NOISEPATH)

    print('create fgnoise...')
    FGNOISEPATH = os.path.join(root_path, 'FGNOISE')
    if not os.path.exists(FGNOISEPATH):
        os.makedirs(FGNOISEPATH)
    smooth_maps(bl_temp_arr, bl_grad_arr, bl_curl_arr, bl_std_temp, bl_std_grad, bl_std_curl, m_cmb_files=None, m_fg_files=fg_data_list,m_nstd_files=nstd_data_list, save_path=FGNOISEPATH, bin_mask=bin_mask, apo_mask=apo_mask)
    split_maps(FGNOISEPATH)

    print('create cmbfg...')
    CMBFGPATH = os.path.join(root_path, 'CMBFG')
    if not os.path.exists(CMBFGPATH):
        os.makedirs(CMBFGPATH)
    smooth_maps(bl_temp_arr, bl_grad_arr, bl_curl_arr, bl_std_temp, bl_std_grad, bl_std_curl, m_cmb_files=cmb_data_list, m_fg_files=fg_data_list,m_nstd_files=None, save_path=CMBFGPATH, bin_mask=bin_mask, apo_mask=apo_mask)
    split_maps(CMBFGPATH)


def noise_simulator(root_path, nstd_data_list, sim_min=0, sim_max=100):
    print('create noise simulation')
    for i in range(sim_min, sim_max):
        print(f'simulation:{i}')
        NOISEPATH = os.path.join(root_path, f'NOISESIM/{i}')
        if not os.path.exists(NOISEPATH):
            os.makedirs(NOISEPATH)
        smooth_maps(bl_temp_arr, bl_grad_arr, bl_curl_arr, bl_std_temp, bl_std_grad, bl_std_curl, m_cmb_files=None, m_fg_files=None,m_nstd_files=nstd_data_list, save_path=NOISEPATH,num_sim=i)
        split_maps(NOISEPATH)



def main1():
    ''' create full sky simulations '''
    noise_simulator('./FULL_PATCH/noPS_northNOI', nstd_data_list=sorted_nstd, sim_min=30, sim_max=50)
    noise_simulator('./FULL_PATCH/noPS_northLOWNOI', nstd_data_list=sorted_nstd, sim_min=40, sim_max=50)

    data_creater('./FULL_PATCH/noPS_northNOI',cmb_data_list=sorted_cmb, fg_data_list=sorted_fgnps, nstd_data_list=sorted_nstd)
    data_creater('./FULL_PATCH/noPS_northLOWNOI',cmb_data_list=sorted_cmb, fg_data_list=sorted_fgnps, nstd_data_list=sorted_nstd)

    data_creater('./FULL_PATCH/PS_northNOI',cmb_data_list=sorted_cmb, fg_data_list=sorted_fgps, nstd_data_list=sorted_nstd)
    data_creater('./FULL_PATCH/PS_northLOWNOI',cmb_data_list=sorted_cmb, fg_data_list=sorted_fgps, nstd_data_list=sorted_nstd)

    noise_simulator('./FULL_PATCH/PS_northNOI', nstd_data_list=sorted_nstd, sim_min=40, sim_max=50)
    noise_simulator('./FULL_PATCH/PS_northLOWNOI', nstd_data_list=sorted_nstd, sim_min=40, sim_max=50)



if __name__=="__main__":

    # check_maps_IQU(sorted_fgps, lmax, nside)
    bl_temp_arr, bl_grad_arr, bl_curl_arr = bl_creater(lmax)

    bin_mask=np.load('../mask/north/BINMASKG.npy')
    apo_mask=np.load('../mask/north/APOMASKC1_5.npy')

    data_creater('./SM2B/noPS_northNOI_Sm_1_bin', cmb_data_list=sorted_cmb, fg_data_list=sorted_fgnps, nstd_data_list=sorted_nstd, bin_mask=bin_mask, apo_mask=bin_mask)

    # cmbfg = glob.glob('./PART_PATCH/noPS_northNOI_apoMask1/CMB/*.npy')
    # sortedcmbfg = sorted(cmbfg, key=lambda x: int(Path(x).stem))
    # check_maps_TEB(maps_pos_list=sortedcmbfg, lmax=lmax, nside=nside)



