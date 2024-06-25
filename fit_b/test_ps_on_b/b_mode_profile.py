import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nside = 2048
npix = hp.nside2npix(nside)
beam = 11
fwhm = np.deg2rad(beam) / 60
sigma = fwhm / (np.sqrt(8*np.log(2)))
m_b = np.zeros(npix)

ctr_ori_lon = 0
ctr_ori_lat = 0

ipix_ctr = hp.ang2pix(theta=ctr_ori_lon, phi=ctr_ori_lat, lonlat=True, nside=nside)
ctr_theta, ctr_phi = hp.pix2ang(nside=nside, ipix=ipix_ctr)
ctr_vec = np.asarray(hp.pix2vec(nside=nside, ipix=ipix_ctr))
ctr_lon, ctr_lat = hp.pix2ang(nside=nside, ipix=ipix_ctr, lonlat=True)
print(f'{ctr_theta=}, {ctr_phi=}, {ctr_vec=}')

vec_theta = np.array((np.cos(ctr_theta)*np.cos(ctr_phi), np.cos(ctr_theta)*np.sin(ctr_phi), -np.sin(ctr_theta)))
print(f'{vec_theta=}')
norm_vec_theta = np.linalg.norm(vec_theta)
print(f'{norm_vec_theta=}')
vec_phi = np.asarray((-np.sin(ctr_phi), np.cos(ctr_phi), 0))
print(f'{vec_phi=}')
norm_vec_phi = np.linalg.norm(vec_phi)
print(f'{norm_vec_phi=}')

# cos_theta = vec_theta @ vec_phi
# print(f'{cos_theta=}')

ipix_disc = hp.query_disc(nside=nside, vec=ctr_vec, radius=3*np.deg2rad(beam)/60)
print(f'{ipix_disc.shape=}')

vec_disc = np.asarray(hp.pix2vec(nside=nside, ipix=ipix_disc))
print(f'{vec_disc.shape=}')

vec_ctr_to_disc = vec_disc.T - ctr_vec
np.set_printoptions(threshold=np.inf)
print(f'{vec_ctr_to_disc.shape=}')

norm_vec_ctr_to_disc = np.linalg.norm(vec_ctr_to_disc, axis=1)
print(f'{norm_vec_ctr_to_disc.shape=}')

normed_vec_ctr_to_disc = vec_ctr_to_disc.T / norm_vec_ctr_to_disc
normed_vec_ctr_to_disc = np.nan_to_num(normed_vec_ctr_to_disc, nan=0)
print(f'{normed_vec_ctr_to_disc.shape=}')

norm_normed_vec_ctr_to_disc = np.linalg.norm(normed_vec_ctr_to_disc, axis=0)
print(f'{norm_normed_vec_ctr_to_disc=}')

cos_theta = normed_vec_ctr_to_disc.T @ vec_theta
cos_phi = normed_vec_ctr_to_disc.T @ vec_phi

xi = np.arctan2(cos_phi, cos_theta)
# xi = np.arctan2(cos_theta, cos_phi)

# cos_2xi = 1 - 2 * sin_xi**2
# sin_2xi = 2 * sin_xi * cos_xi

cos_2xi = np.cos(2*xi)
sin_2xi = np.sin(2*xi)

r_2 = norm_vec_ctr_to_disc**2
r_2_div_sigma = norm_vec_ctr_to_disc**2 / (2*sigma**2)
ps_2phi = np.arctan2(-1,2)
# ps_phi = 0

m = -(sin_2xi * np.cos(ps_2phi) - cos_2xi * np.sin(ps_2phi)) * (1 / r_2) * (np.exp(-r_2_div_sigma) * (1+r_2_div_sigma) - 1)
print(f'{m=}')
m = np.nan_to_num(m, nan=0)

m_b[ipix_disc] = m
hp.gnomview(m_b, rot=[ctr_lon, ctr_lat, 0], xsize=100)
plt.show()

sum_model = np.sum(np.abs(m))
print(f'{sum_model=}')

sum_real = np.load('./sum.npy')
print(f'{sum_real=}')

m_model = m_b / sum_model * sum_real
m_real = np.load('./m_p_b.npy')

hp.gnomview(m_model, rot=[ctr_lon, ctr_lat, 0], xsize=100, title='model')
hp.gnomview(m_real, rot=[ctr_lon, ctr_lat, 0], xsize=100, title='real')
hp.gnomview(m_real - m_model, rot=[ctr_lon, ctr_lat, 0], xsize=100, title='res')
plt.show()





