import numpy as np
import healpy as hp
import os
import matplotlib.pyplot as plt

freq = 40
beam = 63
lmax = 400

# if not os.path.exists(f'./{freq}GHz'):
#     os.makedirs(f'./{freq}GHz')
# for i in range(50):
#     m = np.load(f'../../src/cmbsim/cmbdata/realization/{i}.npy')
#     sm = hp.smoothing(m, fwhm=np.deg2rad(beam)/60, lmax=lmax)
#     np.save(f'./{freq}GHz/{i}.npy', sm)


m = np.load(f'../../src/cmbsim/cmbdata/lmax500.npy')
sm = hp.smoothing(m, fwhm=np.deg2rad(beam)/60, lmax=lmax)
np.save(f'./lmax400.npy', sm)


