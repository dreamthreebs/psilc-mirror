import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nside = 512
npix = hp.nside2npix(nside)
beam = 63
sigma = np.deg2rad(beam) / 60 / np.sqrt(8*np.log(2))

nstd = np.load('../../FGSim/NSTDNORTH/40.npy')[0]
noise = nstd * np.random.normal(0,1,(npix))

m = np.zeros(npix)

ipix = hp.ang2pix(nside=nside, theta=0, phi=0, lonlat=True)
vec = hp.pix2vec(nside=nside, ipix=ipix)
print(f'{ipix=}')
# m[ipix] = 1
# hp.gnomview(m)
# plt.show()


ipix_disc = hp.query_disc(nside=nside, vec=vec, radius=10 * np.deg2rad(beam)/60 )
m_val = np.zeros(len(ipix_disc))
norm = 3

for i, idx_pix in enumerate(ipix_disc):
    vec_pix = hp.pix2vec(nside=nside, ipix=idx_pix)
    theta = np.arccos(np.array(vec_pix) @ vec)
    m_val[i] = norm / (2*np.pi*sigma**2) * np.exp(- theta**2 / (2 * sigma**2))


m[ipix_disc] = m_val
m = m + noise

hp.gnomview(m)
plt.show()

np.save('./ps.npy', m)


