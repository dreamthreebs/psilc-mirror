import numpy as np
import healpy as hp
import pandas as pd
import time
import matplotlib.pyplot as plt

from pathlib import Path
from nilc import NILC

freq_list = [30, 95, 155, 215, 270]
beam_list = [67, 30, 17, 11, 9]
beam_base = 17 #arcmin
lmax = 1500
nside = 2048

def calc_lmax():
    for freq, beam in zip(freq_list, beam_list):
        lmax = int(2 * np.pi / np.deg2rad(beam) * 60) + 1
        print(f'{freq=}, {beam=}, {lmax=}')

def collect_diff_freq_maps():
    map_list = []
    for freq in freq_list:
        m = np.load(f'../{freq}GHz/fit_res/sm/mean/cfn/0.npy')
        map_list.append(m)
    map_arr = np.asarray(map_list)
    print(f'{map_arr.shape=}')
    np.save('./test/cfn.npy', map_arr)

def try_nilc():
    sim = np.load('./test/cfn.npy')
    mask = np.load(f'../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_3APO_5.npy')

    # do nilc and save the weights in map
    time0 = time.time()
    obj_nilc = NILC(bandinfo='./band_info.csv', needlet_config='./needlets/0.csv', weights_name='./test/weight/w_map.npz', Sm_maps=sim, mask=mask, lmax=lmax, nside=nside, n_iter=3, weight_in_alm=False)
    clean_map = obj_nilc.run_nilc()
    np.save('./test/cln_cfn.npy', clean_map)
    print(f'{time.time()-time0=}')

def check_try_nilc():
    cln_map = np.load(f'./test/cln_cmb_w_map.npy')
    hp.orthview(cln_map, rot=[100,50,0])
    plt.show()

# calc_lmax()
collect_diff_freq_maps()
try_nilc()
# check_try_nilc()




