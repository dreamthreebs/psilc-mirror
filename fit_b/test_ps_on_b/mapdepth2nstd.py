import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pandas as pd
from astropy import units as u
from astropy.coordinates import Angle

def calc_sigma(Delta, Nside):
    # Calculate pixel area in steradian
    pixel_area_sr = 4 * np.pi / (12 * Nside**2) * u.sr

    # Convert pixel area to arcmin^2
    pixel_area_arcmin2 = pixel_area_sr.to(u.arcmin**2)

    # Calculate sigma
    sigma = Delta / np.sqrt(pixel_area_arcmin2)

    return sigma

df = pd.read_csv('../../FGSim/FreqBand')
n_freq = len(df)
print(f'{n_freq}')

for i in range(n_freq):
    freq = df.at[i,'freq']
    Delta = 1.9 * u.uK * u.arcmin
    Nside = 2048

    sigma = calc_sigma(Delta, Nside)
    print(f"The noise standard deviation (sigma) at {freq=} is: {sigma}")


# Delta = 10 * u.uK * u.arcmin  # Replace 10 with your actual map-depth
# Nside = 128  # Replace 128 with your actual Nside

# sigma = calc_sigma(Delta, Nside)

# print(f"The noise standard deviation (sigma) is: {sigma}")


