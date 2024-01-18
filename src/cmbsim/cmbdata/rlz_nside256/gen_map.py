import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

beam = 63
lmax = 800

for i in range(200):
    m = np.load(f'./{i}.npy')
    sm = hp.smoothing(m, fwhm=np.deg2rad(beam)/60, lmax=800)
    np.save(f'./40GHz/{i}.npy', sm)
