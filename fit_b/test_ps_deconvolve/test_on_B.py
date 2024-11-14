import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt

nside = 1024
npix = hp.nside2npix(nside)

beam = 67
beam_out = 11

lmax_m = 3 * nside - 1
lmax_cl = 700

l = np.arange(lmax_cl + 1)

lon = 0
lat = 0

def gen_ps():
    ipix = hp.ang2pix(theta=lon, phi=lat, nside=nside, lonlat=True)
    delta_Q = 10000
    delta_U = 10000

    m = np.zeros((3,npix))
    m[1,ipix] = delta_Q
    m[2,ipix] = delta_U
    smooth_iqu = hp.smoothing(m, fwhm=np.deg2rad(beam)/60)
    smooth_m = hp.alm2map(hp.map2alm(smooth_iqu, lmax=lmax_m)[2], nside=nside)

    np.save(f'./ps_B_{nside}.npy', smooth_m)

def gen_ps_beam_out():
    ipix = hp.ang2pix(theta=lon, phi=lat, nside=nside, lonlat=True)
    delta_Q = 10000
    delta_U = 10000

    m = np.zeros((3,npix))
    m[1,ipix] = delta_Q
    m[2,ipix] = delta_U
    smooth_iqu = hp.smoothing(m, fwhm=np.deg2rad(beam_out)/60)
    smooth_m = hp.alm2map(hp.map2alm(smooth_iqu, lmax=lmax_m)[2], nside=nside)

    np.save(f'./ps_B_{nside}_{beam_out}.npy', smooth_m)


def see_ps():
    # m = np.load(f'./ps_B_{nside}.npy')
    m = np.load(f'./ps_B_1024_11.npy')
    # m1 = np.load(f'./ps_B_{nside}_{beam_out}.npy')
    # m1 = np.load(f'./ps_1024_11to67.npy')
    m1 = np.load(f'./ps_B_1024_67to11.npy')
    hp.gnomview(m, rot=[lon, lat, 0], xsize=1000)
    hp.gnomview(m1, rot=[lon, lat, 0], xsize=1000)
    plt.show()

def ps_power_spectrum():
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax_cl)
    m = np.load(f'./ps_B_{nside}.npy')
    cl = hp.anafast(m, lmax=lmax_cl)
    plt.loglog(l, l*(l+1)*cl/bl**2/(2*np.pi))
    plt.show()

def deconvolve():
    m = np.load(f'./ps_B_{nside}.npy')

    lmax_m = 700
    bl_out = hp.gauss_beam(fwhm=np.deg2rad(beam_out)/60, lmax=lmax_m)
    bl_in = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax_m)
    deconvolve_m = hp.alm2map(hp.almxfl(hp.map2alm(m, lmax=lmax_m), fl=bl_out/bl_in), nside=nside)
    l = np.arange(lmax_m+1)
    plt.loglog(l, bl_out/bl_in)
    plt.show()

    np.save(f'./ps_B_{nside}_{beam}to{beam_out}.npy', deconvolve_m)

def deconvolve_out():
    m = np.load(f'./ps_{nside}_{beam_out}.npy')

    lmax_m = 3*nside - 1
    bl_out = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax_m)
    bl_in = hp.gauss_beam(fwhm=np.deg2rad(beam_out)/60, lmax=lmax_m)
    deconvolve_m = hp.alm2map(hp.almxfl(hp.map2alm(m, lmax=lmax_m), fl=bl_out/bl_in), nside=nside)
    l = np.arange(lmax_m+1)
    plt.loglog(l, bl_out/bl_in)
    plt.show()

    np.save(f'./ps_{nside}_{beam_out}to{beam}.npy', deconvolve_m)


def see_deconvolve_ps():
    m_deconvolve = np.load(f'./ps_{nside}_{beam}to{beam_out}.npy')
    m = np.load(f'./ps_{nside}.npy')
    hp.gnomview(m, rot=[lon, lat, 0], title='input', xsize=1000)
    hp.gnomview(m_deconvolve, rot=[lon, lat, 0], title='deconvolve', xsize=1000)
    plt.show()

def deconvolve_ps_spectrum():
    m = np.load(f'./ps_B_{nside}.npy')
    m1 = np.load(f'./ps_B_{nside}_{beam}to{beam_out}.npy')
    m2 = np.load(f'./ps_B_1024_11.npy')

    cl_in = hp.anafast(m, lmax=lmax_cl)
    cl_out = hp.anafast(m1, lmax=lmax_cl)
    cl_2 = hp.anafast(m2, lmax=lmax_cl)

    bl_out = hp.gauss_beam(fwhm=np.deg2rad(beam_out)/60, lmax=lmax_cl)
    bl_in = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax_cl)

    plt.loglog(l, l*(l+1)*cl_out/bl_out**2/(2*np.pi), label=f'deconvolve to {beam_out} arcmin')
    plt.loglog(l, l*(l+1)*cl_in/bl_in**2/(2*np.pi), label=f'{beam} arcmin point source')
    plt.loglog(l, l*(l+1)*cl_2/bl_out**2/(2*np.pi), label=f'exactly 11 arcmin')
    plt.legend()
    plt.show()

def gen_apo_mask(disc_radius=3):
    vec = hp.ang2vec(theta=0, phi=0, lonlat=True)
    disc_pix = hp.query_disc(nside=nside, vec=vec, radius=np.deg2rad(disc_radius))
    mask = np.zeros(npix)
    mask[disc_pix] = 1
    apo_mask = nmt.mask_apodization(mask_in=mask, aposize=1)

    hp.gnomview(mask, rot=[lon, lat, 0], title='binary mask', xsize=1000)
    hp.gnomview(apo_mask, rot=[lon, lat, 0], title='apodized mask', xsize=1000)
    plt.show()

    return mask, apo_mask

def check_ps():
    m = np.load(f'./ps_B_{nside}.npy')
    m1 = np.load(f'./ps_B_{nside}_{beam}to{beam_out}.npy')
    m2 = np.load(f'./ps_B_1024_11.npy')

    hp.gnomview(m1, rot=[lon, lat, 0], title='from 67 arcmin to 11 arcmin', xsize=1000)
    hp.gnomview(m2, rot=[lon, lat, 0], title='exactly 11 arcmin', xsize=1000)
    lmax_cl = 700
    l = np.arange(lmax_cl + 1)
    disc_radius = 2
    bin_mask, apo_mask = gen_apo_mask(disc_radius)
    cl_in = hp.anafast(m*apo_mask, lmax=lmax_cl)
    cl_out = hp.anafast(m1*apo_mask, lmax=lmax_cl)
    cl_2 = hp.anafast(m2*apo_mask, lmax=lmax_cl)

    bl_out = hp.gauss_beam(fwhm=np.deg2rad(beam_out)/60, lmax=lmax_cl)
    bl_in = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax_cl)

    plt.loglog(l, l*(l+1)*cl_2/bl_out**2/(2*np.pi), label=f'exactly {beam_out} arcmin, mask in {disc_radius} degree')
    plt.loglog(l, l*(l+1)*cl_out/bl_out**2/(2*np.pi), label=f'deconvolve to {beam_out} arcmin, mask in {disc_radius} degree')
    # plt.loglog(l, l*(l+1)*cl_in/bl_in**2/(2*np.pi), label=f'{beam} arcmin point source')
    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell^{BB}[\mu K^2]$')
    plt.legend()
    plt.show()





# gen_ps()
# gen_ps_beam_out()
# see_ps()
# ps_power_spectrum()
# deconvolve()
# deconvolve_out()
# see_deconvolve_ps()
# deconvolve_ps_spectrum()
# gen_apo_mask()
check_ps()

