import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from numpy.polynomial.legendre import Legendre


def main():
    m = np.load('../../../FGSim/FITDATA/PSCMBFGNOISE/40.npy')[0]
    nstd = np.load('../../../FGSim/NSTDNORTH/2048/40.npy')[0]
    df_mask = pd.read_csv('../../partial_sky_ps/ps_in_mask/mask40.csv')
    flux_idx = 2
    lon = np.rad2deg(df_mask.at[flux_idx, 'lon'])
    lat = np.rad2deg(df_mask.at[flux_idx, 'lat'])
    iflux = df_mask.at[flux_idx, 'iflux']

    df_ps = pd.read_csv('../../../test/ps_sort/sort_by_iflux/40.csv')
    
    lmax = 350
    nside = 2048
    beam = 63
    radius_factor = 0.1
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)
    cl_cmb = np.load('../../../src/cmbsim/cmbdata/cmbcl.npy')[:lmax+1,0]

    cl_cmb = cl_cmb * bl**2

    ctr0_pix = hp.ang2pix(nside=nside, theta=lon, phi=lat, lonlat=True)
    ctr0_vec = np.array(hp.pix2vec(nside=nside, ipix=ctr0_pix)).astype(np.float64)

    ipix_fit = hp.query_disc(nside=nside, vec=ctr0_vec, radius=radius_factor * np.deg2rad(beam) / 60)
    vec_around = np.array(hp.pix2vec(nside=nside, ipix=ipix_fit.astype(int))).astype(np.float64)
    # print(f'{ipix_fit.shape=}')
    ndof = len(ipix_fit)
    n_cov = len(ipix_fit)
    cov = np.zeros((n_cov, n_cov))

    theta = hp.rotator.angdist(dir1=ctr0_vec, dir2=vec_around)
    cos_theta = np.cos(theta)
    # cos_theta = ctr0_vec @ vec_around
    print(f'{cos_theta.shape=}')
    print(f'{cos_theta}')
    print(np.max(cos_theta))

    def calc_Plx(cos_theta):
        Plx_arr = np.zeros(lmax+1)
        for l in np.arange(lmax+1):
            Plx_arr[l] = Legendre.basis(l)(cos_theta)
        # print(f'{Plx_arr=}')
        return Plx_arr

    ell = np.arange(lmax+1)

    theta_cache = {}
    for i in range(n_cov):
        print(f'{i=}')
        for j in range(i+1):
            if i == j:
                cov[i, i] = 1 / (4 * np.pi) * np.sum((2 * ell + 1) * cl_cmb[:lmax+1])
            else:
                ipix_i = ipix_fit[i]
                ipix_j = ipix_fit[j]
                vec_i = hp.pix2vec(nside=nside, ipix=ipix_i)
                vec_j = hp.pix2vec(nside=nside, ipix=ipix_j)
                theta = hp.rotator.angdist(vec_i, vec_j)
                print(f'{type(theta)}')
                cos_theta = np.cos(theta)
                # cos_theta = np.dot(vec_i, vec_j)  # Assuming this results in a single value
                # cos_theta = min(1.0, max(cos_theta, -1.0))  # Ensuring it's within [-1, 1]
                cos_theta = float(cos_theta)
    
                # Use cos_theta as a key in the dictionary
                if cos_theta not in theta_cache:
                    cov_ij = 1 / (4 * np.pi) * np.sum((2 * ell + 1) * cl_cmb * calc_Plx(cos_theta))
                    theta_cache[cos_theta] = cov_ij
                else:
                    cov_ij = theta_cache[cos_theta]
    
                cov[i, j] = cov_ij
                cov[j, i] = cov_ij

    np.save('./cov1', cov)


main()




