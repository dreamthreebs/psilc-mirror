import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from pathlib import Path

lmax = 500
nside = 2048
freq = 30
beam = 67
m_lmax = 600
fg = np.load('../../fitdata/2048/FG/30/fg.npy')
apo_mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_8.npy')
bin_mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')
fsky = np.sum(apo_mask) / np.size(apo_mask)

# masked_fg = fg * apo_mask
# hp.orthview(fg[0], rot=[100,50,0], half_sky=True)
# hp.orthview(masked_fg[0], rot=[100,50,0], half_sky=True)
# plt.show()

bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=m_lmax, pol=True)[:,2]
bl_sca = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=m_lmax, pol=True)[:,0]
# print(f'{bl[2000:2050]=}')
# fg_full_b = hp.alm2map(hp.map2alm(fg, lmax=m_lmax)[2], nside=nside)

def debeam_full_b():
    fg_b = hp.alm2map(hp.almxfl(hp.map2alm(fg, lmax=m_lmax)[2], fl=1/bl), nside=nside)
    fg_e = hp.alm2map(hp.almxfl(hp.map2alm(fg, lmax=m_lmax)[1], fl=1/bl), nside=nside)
    fg_t = hp.alm2map(hp.almxfl(hp.map2alm(fg, lmax=m_lmax)[0], fl=1/bl_sca), nside=nside)

    # hp.orthview(fg_b * apo_mask, rot=[100,50,0], half_sky=True)
    # plt.show()
    # cl_full_fg_b = hp.anafast(fg_full_b*apo_mask, lmax=m_lmax)
    cl_fg_b = hp.anafast(fg_b*apo_mask, lmax=m_lmax) * bl**2 / fsky
    cl_fg_e = hp.anafast(fg_e*apo_mask, lmax=m_lmax) * bl**2 / fsky
    cl_fg_t = hp.anafast(fg_t*apo_mask, lmax=m_lmax) * bl_sca**2 / fsky

    cl_fg_b[0:2] = 0
    cl_fg_e[0:2] = 0
    cl_fg_t[0:2] = 0

    # l = np.arange(m_lmax+1)
    # plt.loglog(l*(l+1)*cl_fg_b/(2*np.pi) / fsky, label='first debeam full sky B map, then add apo mask, estimate power')
    # plt.loglog(l*(l+1)*cl_fg_b*bl**2/(2*np.pi) / fsky, label='first debeam full sky B map, add apo mask, estimate power, then multiply beam on power spectrum level ')
    # plt.loglog(l*(l+1)*cl_full_fg_b/bl**2/(2*np.pi) / fsky, label='first get full sky B map, no debeam, add apo mask, estimate power, then debeam on power spectrum level')

    plt.legend()
    plt.xlabel("$\\ell$")
    plt.ylabel("$D_\\ell^{BB}$")
    plt.show()

    cl_fg = np.array([cl_fg_t, cl_fg_e, cl_fg_b])
    path_data = Path(f'./data_debeam_full_b')
    path_data.mkdir(exist_ok=True, parents=True)
    np.save(path_data / Path('cl_fg.npy'), cl_fg)

# debeam_full_b()

def debeam_full_qu():

    alm_b = hp.almxfl(hp.map2alm(fg, lmax=m_lmax)[2], fl=1/bl)
    alm_e = hp.almxfl(hp.map2alm(fg, lmax=m_lmax)[1], fl=1/bl)
    alm_t = hp.almxfl(hp.map2alm(fg, lmax=m_lmax)[0], fl=1/bl_sca)

    _qu = hp.alm2map([alm_t, alm_e, alm_b], nside=nside)

    cl_fg = hp.anafast(_qu*apo_mask, lmax=m_lmax)
    cl_fg[0] = cl_fg[0] * bl_sca**2 / fsky
    cl_fg[0,0:2] = 0
    cl_fg[1] = cl_fg[1] * bl**2 / fsky
    cl_fg[1,0:2] = 0
    cl_fg[2] = cl_fg[2] * bl**2 / fsky
    cl_fg[2,0:2] = 0
    path_data = Path(f'./data_debeam_full_qu')
    path_data.mkdir(exist_ok=True, parents=True)
    np.save(path_data / Path('cl_fg.npy'), cl_fg)

# debeam_full_qu()

def debeam_partial_b():
    fg_b = hp.alm2map(hp.almxfl(hp.map2alm(fg*bin_mask, lmax=m_lmax)[2], fl=1/bl), nside=nside)
    fg_e = hp.alm2map(hp.almxfl(hp.map2alm(fg*bin_mask, lmax=m_lmax)[1], fl=1/bl), nside=nside)
    fg_t = hp.alm2map(hp.almxfl(hp.map2alm(fg*bin_mask, lmax=m_lmax)[0], fl=1/bl_sca), nside=nside)

    cl_fg_b = hp.anafast(fg_b*apo_mask, lmax=m_lmax) * bl**2 / fsky
    cl_fg_e = hp.anafast(fg_e*apo_mask, lmax=m_lmax) * bl**2 / fsky
    cl_fg_t = hp.anafast(fg_t*apo_mask, lmax=m_lmax) * bl_sca**2 / fsky

    cl_fg_b[0:2] = 0
    cl_fg_e[0:2] = 0
    cl_fg_t[0:2] = 0

    # l = np.arange(m_lmax+1)
    # plt.loglog(l*(l+1)*cl_fg_b/(2*np.pi) / fsky, label='first debeam full sky B map, then add apo mask, estimate power')
    # plt.loglog(l*(l+1)*cl_fg_b*bl**2/(2*np.pi) / fsky, label='first debeam full sky B map, add apo mask, estimate power, then multiply beam on power spectrum level ')
    # plt.loglog(l*(l+1)*cl_full_fg_b/bl**2/(2*np.pi) / fsky, label='first get full sky B map, no debeam, add apo mask, estimate power, then debeam on power spectrum level')

    # plt.legend()
    # plt.xlabel("$\\ell$")
    # plt.ylabel("$D_\\ell^{BB}$")
    # plt.show()

    cl_fg = np.array([cl_fg_t, cl_fg_e, cl_fg_b])
    path_data = Path(f'./data_debeam_partial_b')
    path_data.mkdir(exist_ok=True, parents=True)
    np.save(path_data / Path('cl_fg.npy'), cl_fg)

debeam_partial_b()

def debeam_partial_qu():

    # hp.orthview(fg[1]*bin_mask, rot=[100,50,0], half_sky=True)
    alm_b = hp.almxfl(hp.map2alm(fg*bin_mask, lmax=m_lmax)[2], fl=1/bl)
    alm_e = hp.almxfl(hp.map2alm(fg*bin_mask, lmax=m_lmax)[1], fl=1/bl)
    alm_t = hp.almxfl(hp.map2alm(fg*bin_mask, lmax=m_lmax)[0], fl=1/bl_sca)

    _qu = hp.alm2map([alm_t, alm_e, alm_b], nside=nside, lmax=m_lmax)
    # hp.orthview(_qu[1]*bin_mask, rot=[100,50,0], half_sky=True)
    # plt.show()

    cl_fg = hp.anafast(_qu*apo_mask, lmax=m_lmax)
    cl_fg[0] = cl_fg[0] * bl_sca**2 / fsky
    cl_fg[0,0:2] = 0
    cl_fg[1] = cl_fg[1] * bl**2 / fsky
    cl_fg[1,0:2] = 0
    cl_fg[2] = cl_fg[2] * bl**2 / fsky
    cl_fg[2,0:2] = 0
    path_data = Path(f'./data_debeam_partial_qu')
    path_data.mkdir(exist_ok=True, parents=True)
    np.save(path_data / Path('cl_fg.npy'), cl_fg)

debeam_partial_qu()


