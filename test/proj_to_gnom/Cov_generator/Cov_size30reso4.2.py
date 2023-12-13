import numpy as np
import healpy as hp
import pickle
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("..")
from gnom_proj import GnomProj
os.chdir('..')

beam = 63 # arcmin
lmax = 350
nside = 2048
ps_lon = 0
ps_lat = 0
m = np.load(f'./data/ps_maps/lon0lat0.npy')

bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)
cl = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[:lmax+1,0]
cl = cl * bl**2

size = 30
reso = 4.2
obj = GnomProj(m, lon=ps_lon, lat=ps_lat, xsize=size, ysize=size, reso=reso, nside=2048)
obj.print_init_info()
cov = obj.calc_cov(cl=cl, lmax=lmax)

np.save('./data/cov_size_30_reso4.2.npy', cov)
