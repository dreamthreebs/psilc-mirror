import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

mask_ori = np.load(f'../../../src/mask/north/BINMASKG2048.npy')
fsky_ori = np.sum(mask_ori) / np.size(mask_ori)
print(f'{fsky_ori=}')

mask_C1_5 = np.load(f'../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
fsky_C1_5 = np.sum(mask_C1_5) / np.size(mask_C1_5)
print(f'{fsky_C1_5=}')

mask_out = np.load(f'../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/BIN_C1_5APO_3APO_5.npy')
fsky_out = np.sum(mask_out) / np.size(mask_out)
print(f'{fsky_out=}')

