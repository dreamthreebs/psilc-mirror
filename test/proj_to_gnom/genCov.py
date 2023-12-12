import numpy as np
import healpy as hp
import pickle
import matplotlib.pyplot as plt
from gnom_proj import GnomProj

if __name__ == "__main__":
    beam = 63 # arcmin
    lmax = 350
    nside = 2048
    ps_lon = 0
    ps_lat = 0
    m = np.load(f'./data/ps_maps/lon{ps_lon}lat{ps_lat}.npy')

    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)
    cl = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[:lmax+1,0]
    cl = cl * bl**2

    obj = GnomProj(m, lon=ps_lon, lat=ps_lat, xsize=10, ysize=10, reso=3.0, nside=2048)
    obj.print_init_info()
    cov = obj.calc_cov(cl=cl, lmax=lmax)
    np.save(f'./data/cov_size_10_reso3.npy', cov)
    # obj.test_flatten()



