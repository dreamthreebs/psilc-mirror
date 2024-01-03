import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.integrate import quad

nside = 2048
npix = hp.nside2npix(nside)
lmax = 500
beam = 63
radius_factor = 5
sigma = np.deg2rad(beam) / 60 / (np.sqrt(8 * np.log(2)))
# sigma = np.deg2rad(beam) / 60 / (2.35)

m = np.zeros(npix)

ipix = hp.ang2pix(nside=nside, theta=50, phi=30, lonlat=True)

def mJy_to_uKCMB(intensity_mJy, frequency_GHz):
    # Constants
    c = 2.99792458e8  # Speed of light in m/s
    h = 6.62607015e-34  # Planck constant in J*s
    k = 1.380649e-23  # Boltzmann constant in J/K
    T_CMB = 2.725  # CMB temperature in Kelvin

    # Convert frequency to Hz from GHz
    frequency_Hz = frequency_GHz * 1e9

    # Calculate x = h*nu/(k*T)
    x = (h * frequency_Hz) / (k * T_CMB)
    print(f'{x=}')

    # Calculate the derivative of the Planck function with respect to temperature, dB/dT
    dBdT = (2.0 * h * frequency_Hz**3 / c**2 / T_CMB) * (x * np.exp(x) / (np.exp(x) - 1)**2)

    # Convert intensity from mJy to Jy
    intensity_Jy = intensity_mJy * 1e-3

    # Convert Jy/sr to W/m^2/sr/Hz
    intensity_W_m2_sr_Hz = intensity_Jy * 1e-26

    # Convert to uK_CMB, taking the inverse of dB/dT
    uK_CMB = intensity_W_m2_sr_Hz / dBdT * 1e6

    return uK_CMB / hp.nside2pixarea(nside=2048)

def calc_norm_beam(flux=1e6, radius_factor=5):

    m[ipix] = flux
    sm = hp.smoothing(m, lmax=lmax, fwhm=np.deg2rad(beam)/60)
    
    # hp.gnomview(sm)
    # plt.show()
    print(f'{np.sum(sm)=}')
    # mask = np.zeros(npix)
    
    vec = np.array(hp.pix2vec(nside=nside, ipix=ipix)).astype(np.float64)
    ipix_disc = hp.query_disc(nside=nside, vec=vec, radius=radius_factor * np.deg2rad(beam) / 60)
    # mask[ipix_disc] = 1
    # print(f'{np.sum(mask*sm)=}')
    vec_around = np.array(hp.pix2vec(nside=nside, ipix=ipix_disc.astype(int))).astype(np.float64)
    theta = hp.rotator.angdist(dir1=vec, dir2=vec_around)
    
    def model(sigma, theta):
        return 1 / (2 * np.pi * sigma**2) * np.exp(- (theta)**2 / (2 * sigma**2))
    
    y = model(sigma, theta)
    norm_beam = 1 / (np.sum(y) / flux)
    print(f'{norm_beam=}')
    return norm_beam

def plot_radius_norm_beam_relation():
    norm_beam_list = []
    radius_factor_list = np.linspace(0.5,3,10)
    for r in radius_factor_list:
        norm_beam = calc_norm_beam(radius_factor=r)
        norm_beam_list.append(norm_beam)
    
    plt.plot(radius_factor_list, norm_beam_list)
    plt.xlabel('radius factor for calc norm beam')
    plt.ylabel('norm beam')
    plt.show()

def plot_flux_norm_beam_relation():
    norm_beam_list = []
    flux_list = np.geomspace(1e-3, 1e9, 10)
    
    for flux in flux_list:
        norm_beam = calc_norm_beam(flux=flux)
        norm_beam_list.append(norm_beam)
    
    plt.loglog(flux_list, norm_beam_list)
    plt.xlabel('flux amplitude')
    plt.ylabel('norm beam')
    plt.show()

def plot_proportional_coefficient():
    norm_beam_list = []
    flux_list = np.geomspace(1e-3, 1e9, 10)
    
    for flux in flux_list:
        norm_beam = calc_norm_beam(flux=flux)
        norm_beam_list.append(norm_beam)
    
    plt.plot(flux_list, flux_list/norm_beam_list)
    plt.xlabel('flux amplitude')
    plt.ylabel('proportional coefficient')
    plt.show()

# plot_proportional_coefficient()

def integrand(theta):
    return (1/(2*np.pi*sigma**2)) * np.exp(-theta**2 / (2*sigma**2))

# integral_value, _ = quad(integrand, 0, np.pi)

def see_theo_and_expe():
    flux = 1e6
    norm_beam = calc_norm_beam(flux=flux)
    factor_true = flux / norm_beam
    factor_theo = 1 / hp.nside2pixarea(nside=nside)
    norm_theo = flux / factor_theo
    
    print(f'{flux=}')
    print(f'factor_experiment={factor_true}')
    print(f'factor_theory={factor_theo}')
    print(f'norm_beam experiment={norm_beam}')
    print(f'norm_beam theory={norm_theo}')


intensity_mJy = 11649
frequency_GHz = 40

uK_CMB = mJy_to_uKCMB(intensity_mJy, frequency_GHz)
coefficient = uK_CMB / intensity_mJy
print(f'{coefficient=}')

coeffmJy2norm = 84.9037624542295 /4005184.4286436457 # coeff_mJy2muKcmb * coeff_muKcmb2norm_beam

print(f'{coeffmJy2norm=}')
print(f'{intensity_mJy * coeffmJy2norm=}')



