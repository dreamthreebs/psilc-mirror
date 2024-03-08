import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

for flux_idx in range(136):
    print(f'{flux_idx=}')

    num_ps = np.load(f'./idx_{flux_idx}/num_ps.npy')
    norm_beam = np.load(f'./idx_{flux_idx}/norm_beam.npy')
    norm_error = np.load(f'./idx_{flux_idx}/norm_error.npy')
    fit_lon = np.load(f'./idx_{flux_idx}/fit_lon.npy')
    fit_lat = np.load(f'./idx_{flux_idx}/fit_lat.npy')
    fit_error = np.load(f'./idx_{flux_idx}/fit_error.npy')
    chi2dof = np.load(f'./idx_{flux_idx}/chi2dof.npy')

    nan_count = np.sum(np.isnan(num_ps)) + np.sum(np.isnan(norm_beam)) + np.sum(np.isnan(norm_error)) + np.sum(np.isnan(fit_lon)) + np.sum(np.isnan(fit_lat)) + np.sum(np.isnan(fit_error)) + np.sum(np.isnan(chi2dof))
    print(f'{nan_count=}')
    
    # print(f'{num_ps=}')
    # print(f'{norm_beam=}')
    # print(f'{norm_error=}')
    # print(f'{fit_lon=}')
    # print(f'{fit_lat=}')
    # print(f'{fit_error=}')
    # print(f'{chi2dof=}')




