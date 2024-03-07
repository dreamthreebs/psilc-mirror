import numpy as np

flux_idx = 2
pscmbns_chi2dof = np.load(f'./PSCMBFGNOISE/1.5/idx_{flux_idx}/chi2dof.npy')
pscmbns_norm_beam = np.load(f'./PSCMBFGNOISE/1.5/idx_{flux_idx}/norm_beam.npy')
pscmbns_norm_error = np.load(f'./PSCMBFGNOISE/1.5/idx_{flux_idx}/norm_error.npy')
pscmbns_fit_lon = np.load(f'./PSCMBFGNOISE/1.5/idx_{flux_idx}/fit_lon.npy')
pscmbns_fit_lat = np.load(f'./PSCMBFGNOISE/1.5/idx_{flux_idx}/fit_lat.npy')
pscmbns_fit_error = np.load(f'./PSCMBFGNOISE/1.5/idx_{flux_idx}/fit_error.npy')

print(f'{pscmbns_chi2dof=}')
print(f'{pscmbns_norm_beam=}')
print(f'{pscmbns_norm_error=}')
print(f'{pscmbns_fit_lon=}')
print(f'{pscmbns_fit_lat=}')
print(f'{pscmbns_fit_error=}')

cmbns_chi2dof = np.load(f'./CMBFGNOISE/1.5/idx_{flux_idx}/chi2dof.npy')
cmbns_norm_beam = np.load(f'./CMBFGNOISE/1.5/idx_{flux_idx}/norm_beam.npy')
cmbns_norm_error = np.load(f'./CMBFGNOISE/1.5/idx_{flux_idx}/norm_error.npy')
cmbns_fit_lon = np.load(f'./CMBFGNOISE/1.5/idx_{flux_idx}/fit_lon.npy')
cmbns_fit_lat = np.load(f'./CMBFGNOISE/1.5/idx_{flux_idx}/fit_lat.npy')
cmbns_fit_error = np.load(f'./CMBFGNOISE/1.5/idx_{flux_idx}/fit_error.npy')

print(f'{cmbns_chi2dof=}')
print(f'{cmbns_norm_beam=}')
print(f'{cmbns_norm_error=}')
print(f'{cmbns_fit_lon=}')
print(f'{cmbns_fit_lat=}')
print(f'{cmbns_fit_error=}')

print(f'{pscmbns_norm_beam-cmbns_norm_beam=}')

cmbns_num_ps = np.load(f'./CMBNOISE/1.5/idx_{flux_idx}/num_ps.npy')
print(f'{cmbns_num_ps=}')
