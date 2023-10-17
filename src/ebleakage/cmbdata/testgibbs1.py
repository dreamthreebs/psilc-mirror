import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

# Parameters
nside = 16
lmax = 3 * nside - 1
npix = hp.nside2npix(nside)

# Simulate a sky with random alm (a_{l,m}) coefficients
alm = np.random.randn(hp.Alm.getsize(lmax)) + 1j * np.random.randn(hp.Alm.getsize(lmax))
map_sky = hp.alm2map(alm, nside, lmax=lmax, verbose=False)

# Create a mask: 1 for theta < pi/4, 0 otherwise
theta, phi = hp.pix2ang(nside, np.arange(npix))
mask = np.ones(npix)
mask[theta > np.pi / 4] = 0

# Apply the mask to the sky
map_masked = map_sky * mask

# Compute the power spectrum of the original and masked sky
cl_sky = hp.anafast(map_sky, lmax=lmax)
cl_masked = hp.anafast(map_masked, lmax=lmax)

# Calculate the amplitude of mode-mixing
delta_cl = np.abs(cl_masked - cl_sky)

# Plot the power spectra and mode-mixing
plt.figure()
plt.semilogy(cl_sky, label='Original Sky')
plt.semilogy(cl_masked, label='Masked Sky')
plt.semilogy(delta_cl, label='Mode-Mixing Amplitude', linestyle='--')
plt.xlabel('l')
plt.ylabel('C_l / Delta C_l')
plt.legend()
plt.title('Power Spectrum and Mode-Mixing')
plt.show()

