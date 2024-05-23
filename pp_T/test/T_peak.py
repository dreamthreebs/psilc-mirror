import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

def mJy_to_uKCMB(intensity_mJy, frequency_GHz):
    # Constants
    c = 2.99792458e8  # Speed of light in m/s
    h = 6.62607015e-34  # Planck constant in J*s
    k = 1.380649e-23  # Boltzmann constant in J/K
    T_CMB = 2.725  # CMB temperature in Kelvin

    frequency_Hz = frequency_GHz * 1e9 # Convert frequency to Hz from GHz

    x = (h * frequency_Hz) / (k * T_CMB) # Calculate x = h*nu/(k*T)

    # Calculate the derivative of the Planck function with respect to temperature, dB/dT
    dBdT = (2.0 * h * frequency_Hz**3 / c**2 / T_CMB) * (x * np.exp(x) / (np.exp(x) - 1)**2)
    intensity_Jy = intensity_mJy * 1e-3 # Convert intensity from mJy to Jy
    intensity_W_m2_sr_Hz = intensity_Jy * 1e-26 # Convert Jy/sr to W/m^2/sr/Hz
    uK_CMB = intensity_W_m2_sr_Hz / dBdT * 1e6 # Convert to uK_CMB, taking the inverse of dB/dT
    return uK_CMB

df = pd.read_csv('../../FGSim/FreqBand')
freq = df.at[0,'freq']
beam = df.at[0,'beam']
print(f'{freq=}, {beam=}')

flux_idx = 1

df_ps = pd.read_csv(f'../mask/mask_csv/{freq}.csv')

lon = np.rad2deg(df_ps.at[flux_idx, 'lon'])
lat = np.rad2deg(df_ps.at[flux_idx, 'lat'])
iflux = df_ps.at[flux_idx, 'iflux']

print(f'{lon=}, {lat=}, {iflux=}')

m = np.load(f'../../fitdata/2048/PS/{freq}/ps.npy')[0]

hp.gnomview(m, rot=[lon, lat, 0])
plt.show()

Sm = 202 * 1 / mJy_to_uKCMB(1, 30) * np.pi * ((np.deg2rad(beam)/60)**2) / (4 * np.log(2))

print(f'T_peak = 202, {Sm=}, true_iflux = {iflux}')




