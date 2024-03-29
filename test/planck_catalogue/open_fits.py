from astropy.io import fits
import numpy as np

hdul = fits.open('./COM_PCCS_044_R2.04.fits')
print(repr(hdul[1].header))

for i in np.arange(934):
    psf_flux = hdul[1].data[i]['PSFFLUX']
    psf_flux_err = hdul[1].data[i]['PSFFLUX_ERR']
    psf_ratio = psf_flux_err / psf_flux

    det_flux = hdul[1].data[i]['DETFLUX']
    det_flux_err = hdul[1].data[i]['DETFLUX_ERR']
    det_ratio = det_flux_err / det_flux

    apr_flux = hdul[1].data[i]['APERFLUX']
    apr_flux_err = hdul[1].data[i]['APERFLUX_ERR']
    apr_ratio = apr_flux_err / apr_flux

    gau_flux = hdul[1].data[i]['GAUFLUX']
    gau_flux_err = hdul[1].data[i]['GAUFLUX_ERR']
    gau_ratio = gau_flux_err / gau_flux

    print(f"index={i}, {psf_flux=}, {psf_flux_err=}, {psf_ratio=}")
    print(f" {det_flux=}, {det_flux_err=}, {det_ratio=}")
    print(f" {apr_flux=}, {apr_flux_err=}, {apr_ratio=}")
    print(f" {gau_flux=}, {gau_flux_err=}, {gau_ratio=}")



