import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from iminuit import Minuit
from pathlib import Path

lmax = 2000
nside = 2048
npix = hp.nside2npix(nside)

beam = 11
sigma = np.deg2rad(beam) / 60 / (np.sqrt(8 * np.log(2)))

lon = 0
lat = 0
ipix_ctr = hp.ang2pix(theta=lon, phi=lat, lonlat=True, nside=nside)
pix_lon, pix_lat = hp.pix2ang(ipix=ipix_ctr, nside=nside, lonlat=True)
print(f"{pix_lon=}, {pix_lat=}")
ctr_vec = np.array(hp.pix2vec(nside=nside, ipix=ipix_ctr))
ipix_fit = hp.query_disc(nside=nside, vec=ctr_vec, radius=3 * np.deg2rad(beam) / 60)
ndof = np.size(ipix_fit)
print(f'{ipix_fit.shape=}, {ndof=}')

vec_around = np.array(hp.pix2vec(nside=nside, ipix=ipix_fit.astype(int))).astype(np.float64)
theta = hp.rotator.angdist(dir1=ctr_vec, dir2=vec_around)

def calc_inv_cov():
    cov = np.zeros((ndof, ndof))
    nstd = np.load('../../FGSim/NSTDNORTH/2048/215.npy')[0]
    nstd2 = (nstd**2)[ipix_fit]
    for i in range(ndof):
        cov[i,i] = cov[i,i] + nstd2[i]

    inv_cov = np.linalg.inv(cov)
    return inv_cov

m = np.load('./data/sim.npy')
m_T = m[0].copy()
inv_cov = calc_inv_cov()
print(f'{inv_cov=}')


def lsq_2_params(norm_beam, const):
    def model():
        return norm_beam / (2 * np.pi * sigma**2) * np.exp(- (theta)**2 / (2 * sigma**2)) + const

    y_model = model()
    y_data = m_T[ipix_fit]
    y_diff = y_data - y_model

    z = (y_diff) @ inv_cov @ (y_diff)
    return z


params = (0, 0.0)
obj_minuit = Minuit(lsq_2_params, name=("norm_beam", "const"), *params)
obj_minuit.limits = [(-1e4,1e4), (-1000,1000)]
print(obj_minuit.migrad())
print(obj_minuit.hesse())
chi2dof = obj_minuit.fval / ndof
str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {ndof} = {chi2dof}"
print(str_chi2)



if __name__ == "__main__":
    pass





