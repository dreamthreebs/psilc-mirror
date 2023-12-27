from astropy import units as u
from astropy.cosmology import Planck15
import numpy as np
import healpy as hp
nside = 2048



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

    # return uK_CMB / hp.nside2pixarea(nside=2048)
    return uK_CMB

freq = 40 * u.GHz
equiv = u.thermodynamic_temperature(freq, Planck15.Tcmb0)
print((1. * u.mK).to(u.mJy / u.sr, equivalencies=equiv)*10**-3)

intensity_mJy = 1  # Replace with your value
frequency_GHz = 40  # Replace with your value

uK_CMB = mJy_to_uKCMB(intensity_mJy, frequency_GHz)

print(f'{1/uK_CMB}')



