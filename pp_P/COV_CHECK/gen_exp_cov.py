import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from pathlib import Path

def gen_cov():
    nside = 64
    lmax = 3 * 64 - 1
    l_range = np.arange(2,lmax)
    Cl = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')
    Cl_TT = Cl[:lmax,0]
    Cl_EE = Cl[:lmax,1]
    Cl_BB = Cl[:lmax,2]
    Cl_TE = Cl[:lmax,3]

    m_list_T = []
    m_list_QU = []
    path_exp = Path(f'./{nside}/exp')
    path_exp.mkdir(exist_ok=True, parents=True)

    path_maps = Path(f'./{nside}/maps')
    path_maps.mkdir(exist_ok=True, parents=True)

    for i in range(100):
        print(f'{i=}')
        m = hp.synfast([Cl_TT, Cl_EE, Cl_BB, Cl_TE], lmax=lmax, new=True, pol=True, nside=nside)
        np.save(f'./{nside}/maps/{i}.npy', m)
        m_list_T.append(m[0].copy())
        QU = np.concatenate([m[1].copy(), m[2].copy()])

        m_list_QU.append(QU.copy())

    T_arr = np.array(m_list_T)
    QU_arr = np.array(m_list_QU)
    print(f'{T_arr.shape=}')
    print(f'{QU_arr.shape=}')

    exp_T_cov = np.cov(T_arr, rowvar=False)
    exp_QU_cov = np.cov(QU_arr, rowvar=False)
    print(f'{exp_T_cov.shape=}')
    print(f'{exp_QU_cov.shape=}')

    np.save(path_exp / Path('cov_qu.npy'), exp_QU_cov)

    # hp.mollview(m[0])
    # hp.mollview(m[1])
    # hp.mollview(m[2])
    # plt.show()

gen_cov()

