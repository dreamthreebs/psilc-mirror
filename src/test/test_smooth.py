import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


nside = 2048
# lmax = 3 * nside -1
# lmax = 4 * nside
lmax = 1000

m = np.zeros(hp.nside2npix(nside))

ipix = hp.ang2pix(nside, theta=np.pi/2, phi=0)
print(f'{ipix=}')

m[ipix] = 100

hp.gnomview(m)
# # hp.projscatter(theta=np.pi/2, phi=0)
plt.show()

m2 = hp.smoothing(m, fwhm=np.deg2rad(1), lmax=lmax, iter=5, use_pixel_weights=True)
m3 = hp.smoothing(m, fwhm=np.deg2rad(1.5), lmax=lmax, iter=5, use_pixel_weights=True)
m1 = hp.smoothing(m, fwhm=np.deg2rad(30)/60, lmax=lmax, iter=5, use_pixel_weights=True)
hp.gnomview(m2, title='smooth to 1 degree')
hp.gnomview(m1, title='smooth to 10 arcmin')
hp.gnomview(m3, title='smooth to 1.5 arcmin')
plt.show()

bl = hp.gauss_beam(fwhm=np.deg2rad(1), lmax=lmax)
bl3 = hp.gauss_beam(fwhm=np.deg2rad(1.5), lmax=lmax)
bl1 = hp.gauss_beam(fwhm=np.deg2rad(30)/60, lmax=lmax)

m2_3 = hp.alm2map(hp.almxfl(hp.map2alm(m2, lmax=lmax, use_pixel_weights=True), bl3/bl), nside=nside)
m2_1 = hp.alm2map(hp.almxfl(hp.map2alm(m2, lmax=lmax, use_pixel_weights=True), bl1/bl), nside=nside)

hp.gnomview(m2_1, title='smooth 1 degree to 10 arcmin')
hp.gnomview(m2_3, title='smooth 1 degree to 90 arcmin')
plt.show()

vec = hp.ang2vec(theta=np.pi/2, phi=0)
mask = np.ones(hp.nside2npix(nside))
mask_ipix = hp.query_disc(nside=nside, vec=vec, radius=np.deg2rad(30)/60)
mask[mask_ipix] = 0
hp.gnomview(mask, title='mask fwhm = 30 arcmin disc')

masked_m2 = m2 * mask
hp.gnomview(masked_m2, title='masked 1 degree')
masked_m2_3 = m2_3 * mask
masked_m2_1 = m2_1 * mask
masked_m1 = m1 * mask

hp.gnomview(masked_m2_1, title='masked 1 degree to 10 arcmin')
hp.gnomview(masked_m2_3, title='masked 1 degree to 1.5 degree')
hp.gnomview(masked_m1, title='masked 10 arcmin')
hp.gnomview(masked_m1-masked_m2_1, title='diff between directly smooth2 10 arcmin and 1degree smooth to 10 arcmin' )
plt.show()


cl = hp.anafast(m, lmax=lmax)
cl2 = hp.anafast(m2, lmax=lmax)
cl2_3 = hp.anafast(m2_3, lmax=lmax)
cl2_1 = hp.anafast(m2_1, lmax=lmax)
cl3 = hp.anafast(m3, lmax=lmax)
cl1 = hp.anafast(m1, lmax=lmax)
l = np.arange(lmax+1)

plt.semilogy(l*(l+1)*cl/(2*np.pi), label='cl 1 degree')
plt.semilogy(l*(l+1)*cl2/(2*np.pi), label='cl 1.5 degree')
plt.semilogy(l*(l+1)*cl2_3/(2*np.pi), label='cl 1 degree to 1.5 degree')
plt.semilogy(l*(l+1)*cl2_1/(2*np.pi), label='cl 1 degree to 10 arcmin')
plt.semilogy(l*(l+1)*cl3/(2*np.pi), label='cl 1.5 degree')
plt.semilogy(l*(l+1)*cl1/(2*np.pi), label='cl 10 arcmin')
plt.legend()
plt.show()






