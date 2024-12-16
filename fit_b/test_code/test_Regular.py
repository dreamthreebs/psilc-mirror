import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

def dNdx_exp(x, eps_f, cas_order='n=1'):
    #x_value = ene/mx #energy expressed in terms of x= E/m_DM
    log10x = np.log10(x)
    spec_data = pd.read_table('./Cascade_B_gammas.dat', sep="\s+", usecols=['EpsF', 'Log[10,x]', cas_order])
    print(f'{spec_data=}')
    epfs_fs = np.unique(spec_data['EpsF'])[::-1]
    print(f'{epfs_fs=}')
    log10xs = np.unique(spec_data['Log[10,x]'])
    print(f'{log10xs=}')
    SMspectrum = np.array(spec_data[cas_order]).reshape(len(epfs_fs), len(log10xs), order='C')
    print(f'{SMspectrum=}')
    interpolated_spectrum = RegularGridInterpolator((log10xs, epfs_fs), SMspectrum.T, method='linear', bounds_error=False, fill_value=None)#, fill_value=0)
    eps = interpolated_spectrum((log10x, eps_f))/(np.log(10)*x)
    print(f'{eps=}')

    return eps

dNdx_exp(x=3, eps_f=10)
