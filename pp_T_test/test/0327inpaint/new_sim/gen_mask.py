import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nside = 2048
lmax = 2000
npix = hp.nside2npix(nside)
beam = 17

mask = np.ones(npix)
idx = np.load('./source_indices.npy')
print(f'{idx=}')

vec = np.array(hp.pix2vec(nside=nside, ipix=idx))
print(f'{vec.shape=}')

for i in range(100):
    print(f"{i=}")
    ipix_disc = hp.query_disc(nside=nside, vec=vec[:,i], radius=1.5 * np.deg2rad(beam) / 60)
    print(f'{ipix_disc.shape=}')
    mask[ipix_disc] = 0

# np.save('mask2.npy', mask)
hp.write_map('./mask/mask.fits',mask)
hp.mollview(mask)
plt.show()
