import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from numpy.polynomial import Legendre
from scipy.interpolate import interp1d
import time

def legendre_polynomial(l, x):
    """
    Evaluate the Legendre polynomial of degree l at point x.

    :param l: Degree of the Legendre polynomial.
    :param x: Point at which to evaluate the polynomial.
    :return: Value of the Legendre polynomial of degree l at x.
    """
    # Coefficients for the Legendre polynomial of degree l
    # The coefficient for the l-th term is 1, all others are 0
    coeffs = [0]*(l) + [1]
    
    # Create a Legendre series from the coefficients
    leg_poly = Legendre(coeffs)

    # Evaluate the polynomial at x
    return leg_poly(x)

def calc_theta_near(nside, ipix1, ipix2):
    vec1 = hp.pix2vec(nside=nside, ipix=ipix1)
    vec2 = hp.pix2vec(nside=nside, ipix=ipix2)
    cos_theta_near = np.array(vec1) @ np.array(vec2)
    print(f'{nside=}, {ipix1=}, {ipix2=}, {cos_theta_near=}')

def see_theta_polys():
    l = 500
    print(legendre_polynomial(l, x=cos_theta[0]))

    cos_theta_true = np.linspace(np.min(cos_theta), np.max(cos_theta), 1000)
    lgd = [legendre_polynomial(l, x=theta) for theta in cos_theta]
    lgd_true = [legendre_polynomial(l, x=theta) for theta in cos_theta_true]

    plt.scatter(cos_theta, lgd, label='exp', s=10)
    plt.plot(cos_theta_true, lgd_true, label='true')
    plt.legend()
    # plt.plot(np.sort(cos_theta))
    plt.show()




if __name__=='__main__':
    nside = 2048
    # calc_theta_near(nside=nside, ipix1=0, ipix2=1)

    center_disc_pix = hp.ang2pix(nside=nside, theta=0, phi=0, lonlat=True)
    center_disc_vec = hp.pix2vec(nside=nside, ipix=center_disc_pix)
    print(f'{center_disc_vec=}')

    ipix_disc = hp.query_disc(nside=nside, vec=center_disc_vec, radius=np.deg2rad(1.3))
    vec_around = hp.pix2vec(nside=nside, ipix=ipix_disc)
    cos_theta = np.array(center_disc_vec) @ np.array(vec_around)
    print(f'{cos_theta.shape=}')
    cos_theta_set = set(cos_theta)
    print(f'{len(cos_theta_set)=}')

    # see_theta_polys()
    l = 3000

    # x = np.sort(cos_theta_set)
    print(f'{np.max(cos_theta)}')
    print(f'{np.min(cos_theta)}')

    x_cos_theta = np.linspace(np.min(cos_theta), 1.0, 1000)
    x_cos_theta_geom = np.geomspace(np.min(cos_theta), 1.0, 100)

    time0 = time.time()
    y = [legendre_polynomial(l, x=x) for x in x_cos_theta]
    time1 = time.time()
    time_calc = time1-time0
    print(f'calc time={time_calc}')


    time2 = time.time()
    cubic_interp = interp1d(x_cos_theta, y=y, kind='cubic')
    time3 = time.time()
    time_interp = time3-time2
    print(f'interp time={time_interp}')


    time4 = time.time()
    y_new = cubic_interp(cos_theta)
    time5 = time.time()
    time_data = time5-time4
    print(f'data time={time_data}')

    print(f'{y_new.shape=}')
    plt.plot(np.arange(1000), x_cos_theta,  label='linear')
    plt.scatter(np.arange(100), x_cos_theta_geom, s=10, label='geom')
    # plt.plot(x_cos_theta, y, label='interp1d')
    # plt.scatter(cos_theta, y_new, s=10, label='data')
    plt.legend()
    plt.show()




