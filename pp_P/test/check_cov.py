import numpy as np
import healpy as hp

my_cov = np.load('../../pp_T/270/cmb_cov_2048/r_1.5/1.npy')
ly_cov = np.load('./Cov_T.npy')

cl_cmb = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[0:10,0]
print(f'{cl_cmb=}')

print(f'{my_cov=}')
print(f'{ly_cov=}')
print(f'{my_cov-ly_cov=}')
print(f'{np.max(np.abs(my_cov-ly_cov))=}')

