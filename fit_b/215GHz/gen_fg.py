import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from config import nside, beam, freq

lmax = 2500
m_lmax = 3000
df = pd.read_csv('../../FGSim/FreqBand')
fg = np.load(f'../../fitdata/2048/FG/{freq}/fg.npy')
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

def full_b():

    fg_b = hp.alm2map(hp.map2alm(fg, lmax=m_lmax)[2], nside=nside)
    fg_e = hp.alm2map(hp.map2alm(fg, lmax=m_lmax)[1], nside=nside)
    fg_t = hp.alm2map(hp.map2alm(fg, lmax=m_lmax)[0], nside=nside)

    cl_fg_b = hp.anafast(fg_b, lmax=m_lmax)
    cl_fg_e = hp.anafast(fg_e, lmax=m_lmax)
    cl_fg_t = hp.anafast(fg_t, lmax=m_lmax)

    cl_fg_b[0:2] = 0
    cl_fg_e[0:2] = 0
    cl_fg_t[0:2] = 0

    cl_fg_b = cl_fg_b[:lmax+1]
    cl_fg_e = cl_fg_e[:lmax+1]
    cl_fg_t = cl_fg_t[:lmax+1]

    cl_fg = np.array([cl_fg_t, cl_fg_e, cl_fg_b])
    path_data = Path(f'./data/full_b')
    path_data.mkdir(exist_ok=True, parents=True)
    np.save(path_data / Path('cl_fg.npy'), cl_fg)

# full_b()

def no_debeam_full_b():
    fg_b = hp.alm2map(hp.map2alm(fg, lmax=m_lmax)[2], nside=nside)
    fg_e = hp.alm2map(hp.map2alm(fg, lmax=m_lmax)[1], nside=nside)
    fg_t = hp.alm2map(hp.map2alm(fg, lmax=m_lmax)[0], nside=nside)

    cl_fg_b = hp.anafast(fg_b*apo_mask, lmax=m_lmax) / fsky
    cl_fg_e = hp.anafast(fg_e*apo_mask, lmax=m_lmax) / fsky
    cl_fg_t = hp.anafast(fg_t*apo_mask, lmax=m_lmax) / fsky

    cl_fg_b[0:2] = 0
    cl_fg_e[0:2] = 0
    cl_fg_t[0:2] = 0

    cl_fg_b = cl_fg_b[:lmax+1]
    cl_fg_e = cl_fg_e[:lmax+1]
    cl_fg_t = cl_fg_t[:lmax+1]

    cl_fg = np.array([cl_fg_t, cl_fg_e, cl_fg_b])
    path_data = Path(f'./data/no_debeam_full_b')
    path_data.mkdir(exist_ok=True, parents=True)
    np.save(path_data / Path('cl_fg.npy'), cl_fg)

# no_debeam_full_b()

def no_debeam_full_qu():

    cl_fg = hp.anafast(fg*apo_mask, lmax=m_lmax)
    cl_fg[0] = cl_fg[0] / fsky
    cl_fg[0,0:2] = 0
    cl_fg[1] = cl_fg[1] / fsky
    cl_fg[1,0:2] = 0
    cl_fg[2] = cl_fg[2] / fsky
    cl_fg[2,0:2] = 0

    cl_fg_new = np.empty((3, lmax+1))
    cl_fg_new[0] = cl_fg[0,:lmax+1]
    cl_fg_new[1] = cl_fg[1,:lmax+1]
    cl_fg_new[2] = cl_fg[2,:lmax+1]



    path_data = Path(f'./data/no_debeam_full_qu')
    path_data.mkdir(exist_ok=True, parents=True)
    np.save(path_data / Path('cl_fg.npy'), cl_fg_new)

# no_debeam_full_qu()

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

    cl_fg_b = cl_fg_b[:lmax+1]
    cl_fg_e = cl_fg_e[:lmax+1]
    cl_fg_t = cl_fg_t[:lmax+1]

    # l = np.arange(m_lmax+1)
    # plt.loglog(l*(l+1)*cl_fg_b/(2*np.pi) / fsky, label='first debeam full sky B map, then add apo mask, estimate power')
    # plt.loglog(l*(l+1)*cl_fg_b*bl**2/(2*np.pi) / fsky, label='first debeam full sky B map, add apo mask, estimate power, then multiply beam on power spectrum level ')
    # plt.loglog(l*(l+1)*cl_full_fg_b/bl**2/(2*np.pi) / fsky, label='first get full sky B map, no debeam, add apo mask, estimate power, then debeam on power spectrum level')

    plt.legend()
    plt.xlabel("$\\ell$")
    plt.ylabel("$D_\\ell^{BB}$")
    plt.show()

    cl_fg = np.array([cl_fg_t, cl_fg_e, cl_fg_b])
    path_data = Path(f'./data/debeam_full_b')
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

    cl_fg_new = np.empty((3, lmax+1))
    cl_fg_new[0] = cl_fg[0,:lmax+1]
    cl_fg_new[1] = cl_fg[1,:lmax+1]
    cl_fg_new[2] = cl_fg[2,:lmax+1]

    path_data = Path(f'./data/debeam_full_qu')
    path_data.mkdir(exist_ok=True, parents=True)
    np.save(path_data / Path('cl_fg.npy'), cl_fg_new)

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

    cl_fg_b = cl_fg_b[:lmax+1]
    cl_fg_e = cl_fg_e[:lmax+1]
    cl_fg_t = cl_fg_t[:lmax+1]

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

# debeam_partial_b()

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

# debeam_partial_qu()

def check_fg():
    m_lmax = lmax
    full_b = np.load('./data/full_b/cl_fg.npy')
    no_debeam_full_b = np.load('./data/no_debeam_full_b/cl_fg.npy')
    no_debeam_full_qu = np.load('./data/no_debeam_full_qu/cl_fg.npy')
    debeam_full_b = np.load('./data/debeam_full_b/cl_fg.npy')
    debeam_full_qu = np.load('./data/debeam_full_qu/cl_fg.npy')

    print(f'{full_b.shape=}')
    print(f'{debeam_full_qu.shape=}')

    cl_cmb = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy').T[:,:np.size(debeam_full_qu, axis=1)]
    l = np.arange(np.size(cl_cmb, axis=1))
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=np.size(cl_cmb, axis=1)-1)
    plt.semilogy(l*(l+1)*cl_cmb[2]/(2*np.pi), label='cmb, add beam')

    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')[1]
    npix = hp.nside2npix(nside=nside)
    noise = nstd * np.random.normal(loc=0, scale=1, size=(npix,))
    nl_exp = hp.anafast(noise, lmax=m_lmax)

    map_depth = 1.35
    nl_th = (map_depth/bl)**2 / 3437.748**2

    plt.semilogy(l*(l+1)*nl_exp/bl**2/(2*np.pi), label='noise, exp')
    plt.semilogy(l*(l+1)*nl_th/(2*np.pi), label='noise, theory')

    l = np.arange(m_lmax+1)
    print(f'{l.shape=}')
    print(f'{bl.shape=}')

    plt.figure(1)
    plt.semilogy(l*(l+1)*full_b[2]/bl**2/(2*np.pi), label='full_b')
    plt.semilogy(l*(l+1)*debeam_full_b[2]/bl**2/(2*np.pi), label='debeam_full_b')
    plt.semilogy(l*(l+1)*debeam_full_qu[2]/bl**2/(2*np.pi), label='debeam_full_qu')
    plt.semilogy(l*(l+1)*no_debeam_full_b[2]/bl**2/(2*np.pi), label='no debeam_full_b')
    plt.semilogy(l*(l+1)*no_debeam_full_qu[2]/bl**2/(2*np.pi), label='no debeam_full_qu')
    plt.legend()
    plt.title('deconvolved')
    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell^{BB}$')

    plt.figure(2)
    l = np.arange(m_lmax+1)
    plt.semilogy(l*(l+1)*cl_cmb[2]*bl**2/(2*np.pi), label='cmb, add beam')
    plt.semilogy(l*(l+1)*nl_th*bl**2/(2*np.pi), label='noise, theory')
    plt.semilogy(l*(l+1)*full_b[2]/(2*np.pi), label='full_b')
    plt.semilogy(l*(l+1)*debeam_full_b[2]/(2*np.pi), label='debeam_full_b')
    plt.semilogy(l*(l+1)*debeam_full_qu[2]/(2*np.pi), label='debeam_full_qu')
    plt.semilogy(l*(l+1)*no_debeam_full_b[2]/(2*np.pi), label='no debeam_full_b')
    plt.semilogy(l*(l+1)*no_debeam_full_qu[2]/(2*np.pi), label='no debeam_full_qu')
    plt.legend()
    plt.title('without deconvolved')
    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell^{BB}$')

    plt.show()

def check_cl():
    # full_b = np.load('./data/full_b/cl_fg.npy')
    # no_debeam_full_b = np.load('./data/no_debeam_full_b/cl_fg.npy')
    # no_debeam_full_qu = np.load('./data/no_debeam_full_qu/cl_fg.npy')
    debeam_full_b = np.load('./data/debeam_full_b/cl_fg.npy')
    # debeam_full_qu = np.load('./data/debeam_full_qu/cl_fg.npy')

    cl_cmb = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy').T[:,:np.size(debeam_full_b, axis=1)]
    l = np.arange(np.size(cl_cmb, axis=1))
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=np.size(cl_cmb, axis=1)-1)
    plt.semilogy(l*(l+1)*cl_cmb[2]/(2*np.pi), label='CMB')

    # nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')[1]
    # npix = hp.nside2npix(nside=nside)
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(npix,))
    # nl_exp = hp.anafast(noise, lmax=m_lmax)
    # map_depth = 1.9
    map_depth = df.at[6, 'mapdepth']
    nl_th = (map_depth/bl)**2 / 3437.748**2

    # plt.semilogy(l*(l+1)*nl_exp/bl**2/(2*np.pi), label='noise, exp')
    plt.semilogy(l*(l+1)*nl_th/(2*np.pi), label='Noise')
    # ps = np.load('../../fitdata/2048/PS/215/ps.npy')

    # ps_b = hp.alm2map(hp.map2alm(ps, lmax=m_lmax)[2], nside=nside)
    # ps_e = hp.alm2map(hp.map2alm(ps, lmax=m_lmax)[1], nside=nside)
    # ps_t = hp.alm2map(hp.map2alm(ps, lmax=m_lmax)[0], nside=nside)

    # cl_ps_b = hp.anafast(ps_b*apo_mask, lmax=m_lmax) / fsky
    # cl_ps_e = hp.anafast(ps_e*apo_mask, lmax=m_lmax) / fsky
    # cl_ps_t = hp.anafast(ps_t*apo_mask, lmax=m_lmax) / fsky

    # cl_ps_b[0:2] = 0
    # cl_ps_e[0:2] = 0
    # cl_ps_t[0:2] = 0

    # cl_ps = np.array([cl_ps_t, cl_ps_e, cl_ps_b])


    l = np.arange(lmax+1)
    # plt.semilogy(l*(l+1)*full_b[2]/bl**2/(2*np.pi), label='full_b')
    plt.semilogy(l*(l+1)*debeam_full_b[2]/bl**2/(2*np.pi), label='debeam_full_b')
    # plt.semilogy(l*(l+1)*debeam_full_qu[2]/bl**2/(2*np.pi), label='debeam_full_qu')
    # plt.semilogy(l*(l+1)*no_debeam_full_b[2]/bl**2/(2*np.pi), label='Diffuse fg')
    # plt.semilogy(l*(l+1)*cl_ps[2]/bl**2/(2*np.pi), label='Point source')
    # plt.semilogy(l*(l+1)*no_debeam_full_qu[2]/bl**2/(2*np.pi), label='no debeam_full_qu')
    plt.loglog()
    plt.legend()
    # plt.xlim(2,800)
    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell^{BB}$')
    plt.show()


# full_b()
# no_debeam_full_b()
# no_debeam_full_qu()
debeam_full_b()
# debeam_full_qu()

# check_fg()
check_cl()




