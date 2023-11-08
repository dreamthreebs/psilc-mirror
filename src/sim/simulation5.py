import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from pathlib import Path
from itertools import zip_longest

df = pd.read_csv('../../FGSim/FreqBand5')
n_freq = len(df)
print(f'{n_freq = }')
lmax=500
base_beam = 9 # arcmin
nside=512

cmb_all = glob.glob('../../FGSim/CMB5/*.npy')
sorted_cmb = sorted(cmb_all, key=lambda x: int(Path(x).stem))
print(f'{sorted_cmb}')

fgnps_all = glob.glob('../../FGSim/FG_noPS5/*.npy')
sorted_fgnps = sorted(fgnps_all, key=lambda x: int(Path(x).stem))
print(f'{sorted_fgnps}')

fgps_all = glob.glob('../../FGSim/FG5/*.npy')
sorted_fgps = sorted(fgps_all, key=lambda x: int(Path(x).stem))
print(f'{sorted_fgps}')

nstd_all = glob.glob('../../FGSim/NSTDNORTH5/*.npy')
sorted_nstd = sorted(nstd_all, key=lambda x: int(Path(x).stem))
print(f'{sorted_nstd}')

def create_sim(m_cmb_files=None, m_fg_files=None, m_noise_files=None, m_nstd_files=None, bin_mask=None, apo_mask=None, nside_in=512, nside=512, num_sim=None, save_path=None)->None:
    nside_in = nside_in
    n_pix = hp.nside2npix(nside_in)
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

        if nside!=nside_in: m = hp.ud_grade(m, nside_out=nside) 

        if apo_mask is not None: m = m * apo_mask

        freq = df.at[index, 'freq']

        # np.save(f'./FULL_SKY/SM_lowNOISE/{freq}.npy', smTEB)
        # if num_sim is not None:
            # np.save(f'{save_path}/{num_sim}/{freq}.npy', smTEB)
        # else:
        np.save(f'{save_path}/{freq}.npy', m)

def create_noise(m_nstd_files=None, bin_mask=None, apo_mask=None, nside_in=512, nside=512, num_sim=None, save_path=None)->None:
    nside_in = nside_in
    n_pix = hp.nside2npix(nside_in)
    m_nstd_files = [] if m_nstd_files is None else m_nstd_files

    for index, m_nstd in enumerate(m_nstd_files):

        nstd =np.load(m_nstd) if m_nstd is not None else np.zeros((3,n_pix))
        noise = nstd * np.random.normal(0,1,(3,n_pix))

        m = noise

        if nside!=nside_in: m = hp.ud_grade(m, nside_out=nside) 

        if apo_mask is not None: m = m * apo_mask

        freq = df.at[index, 'freq']

        if not os.path.exists(f'{save_path}/{num_sim}'):
            os.makedirs(f'{save_path}/{num_sim}')
        np.save(f'{save_path}/{num_sim}/{freq}.npy', m)

def data_creater(root_path, cmb_data_list=None, fg_data_list=None, nstd_data_list=None, bin_mask=None, apo_mask=None):
    ''' create all data '''
    print('create cmb...')
    CMBPATH = os.path.join(root_path, 'CMB')
    if not os.path.exists(CMBPATH):
        os.makedirs(CMBPATH)
    create_sim(m_cmb_files=cmb_data_list, m_fg_files=None,m_nstd_files=None, save_path=CMBPATH)

    print('create sim...')
    SIMPATH = os.path.join(root_path, 'SIM')
    if not os.path.exists(SIMPATH):
        os.makedirs(SIMPATH)
    create_sim(m_cmb_files=cmb_data_list, m_fg_files=fg_data_list,m_nstd_files=nstd_data_list, save_path=SIMPATH)

    print('create fg...')
    FGPATH = os.path.join(root_path, 'FG')
    if not os.path.exists(FGPATH):
        os.makedirs(FGPATH)
    create_sim(m_cmb_files=None, m_fg_files=fg_data_list,m_nstd_files=None, save_path=FGPATH)

    print('create noise...')
    NOISEPATH = os.path.join(root_path, 'NOISE')
    if not os.path.exists(NOISEPATH):
        os.makedirs(NOISEPATH)
    create_sim(m_cmb_files=None, m_fg_files=None,m_nstd_files=nstd_data_list, save_path=NOISEPATH)

    print('create fgnoise...')
    FGNOISEPATH = os.path.join(root_path, 'FGNOISE')
    if not os.path.exists(FGNOISEPATH):
        os.makedirs(FGNOISEPATH)
    create_sim(m_cmb_files=None, m_fg_files=fg_data_list,m_nstd_files=nstd_data_list, save_path=FGNOISEPATH)

    print('create cmbfg...')
    CMBFGPATH = os.path.join(root_path, 'CMBFG')
    if not os.path.exists(CMBFGPATH):
        os.makedirs(CMBFGPATH)
    create_sim(m_cmb_files=cmb_data_list, m_fg_files=fg_data_list,m_nstd_files=None, save_path=CMBFGPATH)

def noise_simulator(n_simulation_start, n_simulation_end):
    print('simulate noise')
    for i in range(n_simulation_start, n_simulation_end):
        print(f'simulation:{i}')
        create_noise(m_nstd_files=sorted_nstd, num_sim=i, save_path='./NSIDE512BAND5/NOISESIM')

if __name__ == '__main__':
    # data_creater('./NSIDE512BAND5/PS', cmb_data_list=sorted_cmb, fg_data_list=sorted_fgps, nstd_data_list=sorted_nstd)
    noise_simulator(50,100)




