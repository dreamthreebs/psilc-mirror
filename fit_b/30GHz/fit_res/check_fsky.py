import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os,sys

def calc_fsky_mask():
    bin_mask = np.load('../../../src/mask/north/BINMASKG2048.npy')
    apo_mask = np.load('../../../psfit/fitv4/fit_res/2048/ps_mask/new_mask/apo_C1_3_apo_3_apo_3.npy')
    ps_mask = np.load(f'../inpainting/new_mask/apo_ps_mask.npy')


    hp.orthview(apo_mask, half_sky=True, rot=[100,50,0])
    hp.orthview(ps_mask, half_sky=True, rot=[100,50,0])
    plt.show()

    fsky_apo = np.sum(apo_mask**2) / np.size(apo_mask)
    fsky_ps = np.sum(ps_mask**2) / np.size(ps_mask)

    print(f'{fsky_apo=}')
    print(f'{fsky_ps=}')

def calc_fsky_ps():
    theta_fwhm_arcmin = 60  # example: 30 arcmin beam
    theta_fwhm_rad = np.radians(theta_fwhm_arcmin / 60)  # convert to radians
    R = 1.5 * theta_fwhm_rad  # disc radius

    N_src = 1  # number of sources

    omega_cap = 2 * np.pi * (1 - np.cos(R))  # solid angle of one cap
    omega_tot = N_src * omega_cap  # total area

    fsky_ps = omega_tot / (4 * np.pi)  # sky fraction
    print(f"f_sky from point sources: {fsky_ps:.6f}")

def calc_fsky_disc():
    nside = 2048
    npix = hp.nside2npix(nside)
    fwhm = 60.0
    radius = 1.5 * np.deg2rad(fwhm) / 60
    mask = np.ones(npix)
    vec = hp.ang2vec(theta=0, phi=0, lonlat=True)
    disc_pix = hp.query_disc(nside=nside, vec=vec, radius=radius)
    mask[disc_pix] = 0
    # hp.mollview(mask)
    # plt.show()
    fsky = np.sum(mask**2) / np.size(mask)
    print(f'{fsky=}')
    return mask

def calc_fsky_apo():
    import pymaster as nmt
    mask = calc_fsky_disc()
    apo_mask = nmt.mask_apodization(mask_in=mask, aposize=1, apotype="C1")
    fsky = np.sum(apo_mask**2) / np.size(apo_mask)
    print(f'{fsky=}')



calc_fsky_mask()

# calc_fsky_ps()
# calc_fsky_disc()
calc_fsky_apo()
