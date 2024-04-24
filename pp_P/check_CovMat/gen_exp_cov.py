import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

def gen_cov():
    nside = 8
    lmax = 3 * nside - 1
    l_range = np.arange(2,lmax)
    Cl_TT = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[:lmax,0]
    Cl_EE = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[:lmax,1]
    Cl_BB = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[:lmax,2]
    Cl_TE = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[:lmax,3]

    m_list_T = []
    m_list_QU = []
    for i in range(100):
        print(f'{i=}')
        m = hp.synfast([Cl_TT, Cl_EE, Cl_BB, Cl_TE], new=True, pol=True, nside=nside)
        np.save(f'./data/{i}.npy', m)
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

    # np.save('./Exp_T_cov.npy', exp_T_cov)
    # np.save('./Exp_QU_cov.npy', exp_QU_cov)

    # hp.mollview(m[0])
    # hp.mollview(m[1])
    # hp.mollview(m[2])
    # plt.show()

gen_cov()
