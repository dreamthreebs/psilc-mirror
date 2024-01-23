import numpy as np
import healpy as hp
import os
import matplotlib.pyplot as plt

class GenerateData:
    def __init__(self, n_rlz):
        self.n_rlz = n_rlz

    def gen_CMB(self, freq:'GHz', beam:'arcmin'):

        if not os.path.exists('./2048/CMB'):
            os.mkdir('./2048/CMB')

        if not os.path.exists(f'./2048/CMB/{freq}'):
            os.mkdir(f'./2048/CMB/{freq}')

        for i in range(self.n_rlz):
            print(f'{i=}')
            m = np.load(f'../src/cmbsim/cmbdata/m_realization/{i}.npy')
            sm = hp.smoothing(m, fwhm=np.deg2rad(beam) / 60, lmax=1500)
            np.save(f'./2048/CMB/{freq}/{i}.npy', sm)


if __name__ == "__main__":
    n_rlz = 100
    obj = GenerateData(n_rlz=n_rlz)
    obj.gen_CMB(freq=40, beam=63)
