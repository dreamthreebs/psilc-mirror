import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pickle

from numpy.polynomial.legendre import Legendre

lmax = 350
beam = 63


def evaluate_interp_func(l, x, interp_funcs):
    for interp_func, x_range in interp_funcs[l]:
        if x_range[0] <= x <= x_range[1]:
            return interp_func(x)
    raise ValueError(f"x = {x} is out of the interpolation range for l = {l}")

def calc_C_theta_itp(x, lmax, cl, itp_funcs):
    Pl = np.zeros(lmax+1)
    for l in range(lmax+1):
        Pl[l] = evaluate_interp_func(l, x, interp_funcs=itp_funcs)
    ell = np.arange(lmax+1)
    multi =  1 / (4 * np.pi) * (2 * ell + 1) * cl * Pl
    return multi

def calc_C_theta_itp1(x, lmax, cl):
    Pl = np.zeros(lmax+1)
    for l in range(lmax+1):
        Pl[l] = Legendre.basis(l)(x)
    ell = np.arange(lmax+1)
    multi =  1 / (4 * np.pi) * (2 * ell + 1) * cl * Pl
    return multi


with open('../../test/interpolate_cov/lgd_itp_funcs350.pkl','rb') as f:
    loaded_itp_funcs = pickle.load(f)

cl = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[:lmax+1,0]
bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)
cl = cl * bl**2

multi = calc_C_theta_itp(x=0.99, lmax=350, cl=cl, itp_funcs=loaded_itp_funcs)
multi1 = calc_C_theta_itp1(x=0.99, lmax=lmax, cl=cl)

