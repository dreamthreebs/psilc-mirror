import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax = 1000
beam = 63

# m = np.load('../m_realization/0.npy')

bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)
# sm = np.load('../../../../inpaintingdata/CMBREALIZATION/40GHz/0.npy')

def check_sm():
    cl = hp.anafast(m, lmax=lmax)
    
    cl_sm = hp.anafast(sm, lmax=lmax)
    
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)
    
    plt.semilogy(cl[0]*bl**2, label='from origin map')
    plt.semilogy(cl_sm[0], label='from beamed map')
    plt.legend()
    plt.show()


def save_cl():
    for i in range(50):
        m = np.load(f'../m_realization/{i}.npy')

        cl = hp.anafast(m, lmax=lmax)
        print(f'{cl.shape=}')
        np.save(f'./{i}.npy', cl)

save_cl()

