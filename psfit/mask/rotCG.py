import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nside = 2048

r = hp.rotator.Rotator(coord=['C','G'])

theta_cel, phi_cel = np.deg2rad(25), np.deg2rad(150)
theta_gal, phi_gal = r(theta_cel, phi_cel)
theta_gal1 = np.rad2deg(theta_gal)
phi_gal1 = np.rad2deg(phi_gal)
print(f'{theta_gal1=},{phi_gal1=}')
vec1 = hp.ang2vec(theta=theta_gal, phi=phi_gal)
print(f'{vec1=}')

theta_cel, phi_cel = np.deg2rad(70), np.deg2rad(150)
theta_gal, phi_gal = r(theta_cel, phi_cel)
theta_gal2 = np.rad2deg(theta_gal)
phi_gal2 = np.rad2deg(phi_gal)
print(f'{theta_gal2=},{phi_gal2=}')
vec2 = hp.ang2vec(theta=theta_gal, phi=phi_gal)
print(f'{vec2=}')

theta_cel, phi_cel = np.deg2rad(70), np.deg2rad(250)
theta_gal, phi_gal = r(theta_cel, phi_cel)
theta_gal3 = np.rad2deg(theta_gal)
phi_gal3 = np.rad2deg(phi_gal)
print(f'{theta_gal3=},{phi_gal3=}')
vec3 = hp.ang2vec(theta=theta_gal, phi=phi_gal)
print(f'{vec3=}')

theta_cel, phi_cel = np.deg2rad(25), np.deg2rad(250)
theta_gal, phi_gal = r(theta_cel, phi_cel)
theta_gal4 = np.rad2deg(theta_gal)
phi_gal4 = np.rad2deg(phi_gal)
print(f'{theta_gal4=},{phi_gal4=}')
vec4 = hp.ang2vec(theta=theta_gal, phi=phi_gal)
print(f'{vec4=}')

vertices = np.vstack((vec1,vec2,vec3,vec4))
print(f'{vertices=}')

mask_ipix = hp.query_polygon(nside=nside, vertices=vertices)
mask = np.zeros(hp.nside2npix(nside))
mask[mask_ipix] = 1
# hp.mollview(mask)
mask_north = np.load('../../src/mask/north/BINMASKG2048.npy')
hp.orthview(mask, rot=[100,50,0], half_sky=True, title='mask on galactic coordinate')
hp.orthview(mask_north, rot=[100,50,0], half_sky=True, title='mask from celestial coordinate')
plt.show()

fsky = np.sum(mask) / np.size(mask)
print(f'{fsky=}')
fsky = np.sum(mask_north) / np.size(mask_north)
print(f'{fsky=}')



