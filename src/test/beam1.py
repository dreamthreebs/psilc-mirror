import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nside = 512
lmax = 1024
npix = hp.nside2npix(nside)
print(f'{npix}')

m = np.zeros(hp.nside2npix(nside))
# ipix = hp.ang2pix(nside=nside, theta=np.pi/2, phi=0)

# vec = hp.ang2vec(theta=np.pi/2, phi=0 )

# ipix_disc = hp.query_disc(nside=nside, vec=vec, radius=2 * np.deg2rad(30)/60)

# vec_disc = np.array(hp.pix2vec(nside=nside, ipix=ipix_disc))
# print(f'{vec_disc.shape=}')


ipix = np.arange(npix)

vec = np.array(hp.pix2vec(nside=nside, ipix=ipix))
print(f'{vec.shape=}')

ipix_center = hp.ang2pix(nside=nside, theta=np.pi/2, phi=0)
vec_center = np.array(hp.pix2vec(nside=nside, ipix=ipix_center))
print(f'{vec_center.shape=}')

vec_diff = vec_center @ vec
print(f"{vec_diff.shape=}")

theta = np.arccos(vec_diff)
print(f'{theta.shape=}')

sigma = np.deg2rad(10) / 60 / (np.sqrt(8*np.log(2)))
print(f'{sigma=}')

beam = np.exp(-theta**2/(2*sigma**2))
# beam = np.exp(-theta**2/(2*sigma**2))
beam = beam / np.sum(beam)
print(f"{beam.shape=}")


# mask = np.zeros(hp.nside2npix(nside))
# disc_ipix = hp.query_disc(nside=nside, vec=vec, radius=1.5*np.deg2rad(10)/60)
# mask[disc_ipix] = 1


hp.gnomview(beam)
# hp.gnomview(m1)
# hp.gnomview(m1*mask)
plt.show()


