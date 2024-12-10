import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import sys

from fit_qu_no_const import FitPolPS
from config import lmax, freq, nside, beam
from pix_cov_qu import CovCalculator
from pathlib import Path

pcfn = np.load('./pcfn.npy')

def gen_fg_cl():
    cl_fg = np.load('./data/debeam_full_b/cl_fg.npy')
    Cl_TT = cl_fg[0]
    Cl_EE = cl_fg[1]
    Cl_BB = cl_fg[2]
    Cl_TE = np.zeros_like(Cl_TT)
    return np.array([Cl_TT, Cl_EE, Cl_BB, Cl_TE])

def gen_cmb_cl(beam, lmax):
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=10000, pol=True)
    print(f'{bl[0:10,0]=}')
    print(f'{bl[0:10,1]=}')
    print(f'{bl[0:10,2]=}')
    print(f'{bl[0:10,3]=}')
    # cl = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')
    cl = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    print(f'{cl.shape=}')

    Cl_TT = cl[0:lmax+1,0] * bl[0:lmax+1,0]**2
    Cl_EE = cl[0:lmax+1,1] * bl[0:lmax+1,1]**2
    Cl_BB = cl[0:lmax+1,2] * bl[0:lmax+1,2]**2
    Cl_TE = cl[0:lmax+1,3] * bl[0:lmax+1,3]**2
    return np.asarray([Cl_TT, Cl_EE, Cl_BB, Cl_TE])


def calc_lmin_change():
    flux_idx = 0
    df = pd.read_csv(f'./mask/{freq}.csv')

    cl_fg = gen_fg_cl()
    cl_cmb = gen_cmb_cl(beam=beam, lmax=lmax)

    cl_tot = cl_fg + cl_cmb

    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')

    pix_ind = np.load(f'./pix_idx_qu/{flux_idx}.npy')
    path_cov = Path('./cov_lmin')
    path_cov.mkdir(exist_ok=True, parents=True)

    lmin_list = [10,30,50,70,100,150,200,300,400]
    for lmin in lmin_list:
        obj_cov = CovCalculator(nside=nside, lmin=lmin, lmax=lmax, Cl_TT=cl_tot[0], Cl_EE=cl_tot[1], Cl_BB=cl_tot[2], Cl_TE=cl_tot[3], pixind=pix_ind, calc_opt='polarization', out_pol_opt='QU')
        MP = obj_cov.run_calc_cov()
        np.save(path_cov / Path(f'{flux_idx}.npy'), MP)
        obj_fit = FitPolPS(m_q=pcfn[1].copy(), m_u=pcfn[2].copy(), freq=freq, nstd_q=nstd[1].copy(), nstd_u=nstd[2].copy(), flux_idx=flux_idx, df_mask=df, df_ps=df, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, cov_path=path_cov)
        obj_fit.calc_definite_fixed_cmb_cov()
        obj_fit.calc_covariance_matrix(mode='cmb+noise')
        num_ps, chi2dof, fit_P, fit_P_err, fit_phi, fit_phi_err = obj_fit.fit_all(cov_mode='cmb+noise')
        path_chi2 = Path(f'./chi2_lmin')
        path_chi2.mkdir(exist_ok=True, parents=True)
        np.save(path_chi2 / Path(f"{lmin}.npy"), chi2dof)

def calc_lmax_change(lmax):
    flux_idx = 0
    df = pd.read_csv(f'./mask/{freq}.csv')

    cl_fg = gen_fg_cl()
    cl_cmb = gen_cmb_cl(beam=beam, lmax=lmax)

    cl_tot = cl_fg + cl_cmb

    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')

    pix_ind = np.load(f'./pix_idx_qu/{flux_idx}.npy')
    path_cov = Path('./cov_lmax')
    path_cov.mkdir(exist_ok=True, parents=True)
    lmin = 2

    lmax_list = [150,200,350,400,500,700,900,1300]
    for lmax in lmax_list:
        obj_cov = CovCalculator(nside=nside, lmin=lmin, lmax=lmax, Cl_TT=cl_tot[0], Cl_EE=cl_tot[1], Cl_BB=cl_tot[2], Cl_TE=cl_tot[3], pixind=pix_ind, calc_opt='polarization', out_pol_opt='QU')
        MP = obj_cov.run_calc_cov()
        np.save(path_cov / Path(f'{flux_idx}.npy'), MP)
        obj_fit = FitPolPS(m_q=pcfn[1].copy(), m_u=pcfn[2].copy(), freq=freq, nstd_q=nstd[1].copy(), nstd_u=nstd[2].copy(), flux_idx=flux_idx, df_mask=df, df_ps=df, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, cov_path=path_cov)
        obj_fit.calc_definite_fixed_cmb_cov()
        obj_fit.calc_covariance_matrix(mode='cmb+noise')
        num_ps, chi2dof, fit_P, fit_P_err, fit_phi, fit_phi_err = obj_fit.fit_all(cov_mode='cmb+noise')
        path_chi2 = Path(f'./chi2_lmax')
        path_chi2.mkdir(exist_ok=True, parents=True)
        np.save(path_chi2 / Path(f"{lmax}.npy"), chi2dof)

def see_chi2_lmin():
    lmin_list = [2, 10,30,50,70,100,150,200,300,400]
    chi2_list = []
    for lmin in lmin_list:
        chi2 = np.load(f'./chi2_lmin/{lmin}.npy')
        chi2_list.append(chi2)

    print(f'{chi2_list=}')
    plt.plot(lmin_list, chi2_list)
    plt.xlabel('lmin')
    plt.ylabel('$\\chi^2$')
    plt.show()

def see_chi2_lmax():
    lmax_list = [150,200,350,400,500,700,900,1300]
    chi2_list = []
    for lmax in lmax_list:
        chi2 = np.load(f'./chi2_lmax/{lmax}.npy')
        chi2_list.append(chi2)

    print(f'{chi2_list=}')
    plt.plot(lmax_list, chi2_list)
    plt.xlabel('lmax')
    plt.ylabel('$\\chi^2$')
    plt.show()


# lmax = 100
# def modify_lmax():
#     lmax = lmax + 100  # Error: lmax is referenced before assignment
#     print(lmax)


# calc_lmin_change()
# calc_lmax_change(lmax=lmax)

see_chi2_lmin()
see_chi2_lmax()

# modify_lmax()





