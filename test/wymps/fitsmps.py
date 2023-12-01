import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

m = np.load('./psnoise/sm_ps.npy')
nstd = np.load('../../FGSim/NSTDNORTH/40.npy')[0]

nside = 512
beam = 63
sigma_true = np.deg2rad(beam)/60 / (np.sqrt(8*np.log(2)))
lon = 0
lat = 0

def see_true_map():
    # hp.mollview(m, norm='hist');plt.show()

    hp.gnomview(m, rot=[np.rad2deg(lon), np.rad2deg(lat), 0])
    plt.show()

    vec = hp.ang2vec(theta=np.rad2deg(lon), phi=np.rad2deg(lat), lonlat=True)

    ipix_disc = hp.query_disc(nside=nside, vec=vec, radius=np.deg2rad(beam)/60)

    mask = np.ones(hp.nside2npix(nside))
    mask[ipix_disc] = 0

    hp.gnomview(mask, rot=[np.rad2deg(lon), np.rad2deg(lat), 0])
    plt.show()

# see_true_map()

def lsq(lon_bias, lat_bias, norm_beam, const):
    lon_fit = lon + lon_bias
    lat_fit = lat + lat_bias
    new_center_ipix = hp.ang2pix(nside=nside, theta=np.rad2deg(lon_fit), phi=np.rad2deg(lat_fit), lonlat=True)
    new_center_vec = hp.pix2vec(nside=nside, ipix=new_center_ipix)
    ipix_disc = hp.query_disc(nside=nside, vec=new_center_vec, radius=1*np.deg2rad(beam)/60)

    vec_around = hp.pix2vec(nside=nside, ipix=ipix_disc.astype(int))
    theta = np.arccos(np.array(new_center_vec) @ np.array(vec_around))
    theta = np.nan_to_num(theta)

    def model1():
        return norm_beam / (2*np.pi*sigma_true**2) * np.exp(- (theta)**2 / (2 * sigma_true**2)) + const

    def model2():
        m_ps = np.zeros(hp.nside2npix(nside))
        m_ps[new_center_ipix] = norm_beam
        return hp.smoothing(m_ps, fwhm=np.deg2rad(beam)/60, lmax=2000, iter=3)[ipix_disc] + const

    y_model = model1()
    print(f'{y_model.shape=}')

    y_data = m[ipix_disc]
    # print(f'{y_data.shape=}')
    y_data_err = nstd[ipix_disc]

    z = (y_data - y_model) / (y_data_err)
    return np.sum(z**2)


obj_minuit = Minuit(lsq, lon_bias=0, lat_bias=0, norm_beam=1, const=0)
bias_lonlat = np.deg2rad(0.5)
# obj_minuit.limits = [(-bias_lonlat,bias_lonlat),(-bias_lonlat,bias_lonlat),(0,10),(-100,100)]
obj_minuit.limits = [(-0.5,0.5),(-0.5,0.5),(0,10),(-10,10)]
obj_minuit.fixed[0] = True
obj_minuit.fixed[1] = True
# print(obj_minuit.scan(ncall=100))
# obj_minuit.errors = (0.1, 0.2)
print(obj_minuit.migrad())
print(obj_minuit.hesse())
ndof = 265
str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {ndof} = {obj_minuit.fval/ndof}"
print(str_chi2)


