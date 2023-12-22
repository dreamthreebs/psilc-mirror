import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares


nside = 2048
beam = 63
sigma = np.deg2rad(beam)/60 / (np.sqrt(8*np.log(2)))
print(f'{sigma=}')

m = np.zeros(hp.nside2npix(nside))

factor = 1.2
lon1 = - factor * beam / 60 # degree
lat1 = 0
dir1 = (lon1, lat1)
ipix1 = hp.ang2pix(nside=nside, theta=lon1, phi=lat1, lonlat=True)
m[ipix1] = 1e6

lon2 = factor * beam / 60
lat2 = 0
dir2 = (lon2, lat2)

ipix2 = hp.ang2pix(nside=nside, theta=lon2, phi=lat2, lonlat=True)
m[ipix2] = 1e6

ang = hp.rotator.angdist(dir1=dir1, dir2=dir2, lonlat=True)
ang_deg = np.rad2deg(ang)
print(f'{ang_deg=}')

sm_m = hp.smoothing(m, lmax=1000, fwhm=np.deg2rad(63)/60)

# hp.gnomview(sm_m, xsize=400)
# plt.show()

noise = 6.17 * np.random.normal(loc=0, scale=1, size=(hp.nside2npix(nside)))

m = sm_m + noise

center_vec = np.array(hp.pix2vec(nside=nside, ipix=ipix1))
ipix_fit = hp.query_disc(nside=nside, vec=center_vec, radius=1.0 * np.deg2rad(beam)/60)
print(f'{ipix_fit.shape=}')
def fit_model(theta, norm_beam, const):
    beam_profile = norm_beam / (2*np.pi*sigma**2) * np.exp(- (theta)**2 / (2 * sigma**2)) 
    # print(f'{beam_profile=}')
    return beam_profile + const

vec_around = hp.pix2vec(nside=nside, ipix=ipix_fit.astype(int))
print(f'{vec_around=}')
theta = np.arccos(np.array(center_vec) @ np.array(vec_around))
theta = np.clip(theta, -1, 1)

y_arr = m[ipix_fit]
y_err = 6.17 * np.ones_like(ipix_fit)

lsq = LeastSquares(x=theta, y=y_arr, yerror=y_err, model=fit_model)

obj_minuit = Minuit(lsq, norm_beam=1,  const=0)
obj_minuit.limits = [(0,10),(-1e4,1e4)]
# print(obj_minuit.scan(ncall=100))
# obj_minuit.errors = (0.1, 0.2)
print(obj_minuit.migrad())
print(obj_minuit.hesse())





