import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


nside = 2048
# lmax = 3 * nside -1
# lmax = 4 * nside
lmax = 750

m = np.zeros(hp.nside2npix(nside))

ipix = hp.ang2pix(nside, theta=np.pi/2, phi=0)
print(f'{ipix=}')

m[ipix] = 100

hp.gnomview(m)
# # hp.projscatter(theta=np.pi/2, phi=0)
plt.show()

m63 = hp.smoothing(m, fwhm=np.deg2rad(63)/60, lmax=lmax)
m30 = hp.smoothing(m, fwhm=np.deg2rad(30)/60, lmax=lmax)
m17 = hp.smoothing(m, fwhm=np.deg2rad(17)/60, lmax=lmax)
m10 = hp.smoothing(m, fwhm=np.deg2rad(10)/60, lmax=lmax)

hp.gnomview(m63, title='smooth to 63 arcmin')
hp.gnomview(m30, title='smooth to 30 arcmin')
hp.gnomview(m17, title='smooth to 17 arcmin')
hp.gnomview(m10, title='smooth to 10 arcmin')
plt.show()

cl63 = hp.anafast(m63, lmax=lmax)
cl10 = hp.anafast(m10, lmax=lmax)

bl63 = hp.gauss_beam(fwhm=np.deg2rad(63) / 60, lmax=lmax)
bl10 = hp.gauss_beam(fwhm=np.deg2rad(10) / 60, lmax=lmax)

m63_10 = hp.alm2map(hp.almxfl(hp.map2alm(m63, lmax=lmax), bl10/bl63), nside=nside)
cl63_10 = hp.anafast(m63_10, lmax=lmax)

l = np.arange(lmax+1)
plt.semilogy(l*(l+1)*cl63/(2*np.pi), label='cl 63')
plt.semilogy(l*(l+1)*cl10/(2*np.pi), label='cl 10')
# plt.semilogy(l*(l+1)*cl63_10/(2*np.pi), label='cl 63 to 10')
plt.legend()
plt.show()




# vec = hp.ang2vec(theta=np.pi/2, phi=0)
# mask = np.ones(hp.nside2npix(nside))
# mask_ipix = hp.query_disc(nside=nside, vec=vec, radius=np.deg2rad(30)/60)
# mask[mask_ipix] = 0
# hp.gnomview(mask, title='mask fwhm = 30 arcmin disc')

# masked_m2 = m2 * mask
# hp.gnomview(masked_m2, title='masked 1 degree')
# masked_m2_3 = m2_3 * mask
# masked_m2_1 = m2_1 * mask
# masked_m1 = m1 * mask

# hp.gnomview(masked_m2_1, title='masked 1 degree to 10 arcmin')
# hp.gnomview(masked_m2_3, title='masked 1 degree to 1.5 degree')
# hp.gnomview(masked_m1, title='masked 10 arcmin')
# hp.gnomview(masked_m1-masked_m2_1, title='diff between directly smooth2 10 arcmin and 1degree smooth to 10 arcmin' )
# plt.show()


# cl = hp.anafast(m, lmax=lmax)
# cl2 = hp.anafast(m2, lmax=lmax)
# cl2_3 = hp.anafast(m2_3, lmax=lmax)
# cl2_1 = hp.anafast(m2_1, lmax=lmax)
# cl3 = hp.anafast(m3, lmax=lmax)
# cl1 = hp.anafast(m1, lmax=lmax)
# cl2_1_no_weight = hp.anafast(m2_1_no_weight, lmax=lmax)
# l = np.arange(lmax+1)

# plt.semilogy(l*(l+1)*cl/(2*np.pi), label='cl')
# plt.semilogy(l*(l+1)*cl2/(2*np.pi), label='cl 1 degree')
# plt.semilogy(l*(l+1)*cl2_3/(2*np.pi), label='cl 1 degree to 1.5 degree')
# plt.semilogy(l*(l+1)*cl2_1/(2*np.pi), label='cl 1 degree to 10 arcmin')
# plt.semilogy(l*(l+1)*cl3/(2*np.pi), label='cl 1.5 degree')
# plt.semilogy(l*(l+1)*cl1/(2*np.pi), label='cl 10 arcmin')
# plt.semilogy(l*(l+1)*cl2_1_no_weight/(2*np.pi), label='cl 10 arcmin no pixel weight')
# plt.legend()
# plt.show()






