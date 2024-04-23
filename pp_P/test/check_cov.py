import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

def main():

    my_cov = np.load('../../pp_T/270/cmb_cov_2048/r_1.5/1.npy')
    ly_cov = np.load('./Cov_T_another.npy')
    exp_t_cov = np.load('./exp_cov_I.npy')
    
    cl_cmb = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[0:10,0]
    print(f'{cl_cmb=}')
    
    print(f'{exp_t_cov=}')
    print(f'{my_cov=}')
    print(f'{ly_cov=}')
    print(f'{my_cov-ly_cov=}')
    print(f'{np.max(np.abs(my_cov-ly_cov))=}')

def check_qu():
    qu_cov = np.load('./Cov_QU.npy')
    qu_exp = np.load('./exp_cov_QU.npy')
    print(f'{qu_cov=}')
    print(f'{qu_exp=}')

def plot_qu_cov():
    qu_cov = np.load('./Cov_QU.npy')
    # cov_noise = np.load('./semi_def_cmb_cov_2048/r_1.5/1.npy')
    cov_exp = np.load('./exp_cov_QU.npy')
    # plt.imshow(qu_cov, cmap='viridis', interpolation='nearest')
    plt.figure(1)
    cax = plt.imshow(qu_cov, cmap='viridis')
    plt.title('theory')
    plt.colorbar(cax)
    plt.figure(2)
    cax = plt.imshow(cov_exp, cmap='viridis')
    plt.title('exp')
    plt.colorbar(cax)
    plt.figure(3)
    cax = plt.imshow(qu_cov-cov_exp, cmap='viridis')
    plt.title('res')
    plt.colorbar(cax)
    plt.show()
# check_qu()
# main()
plot_qu_cov()
