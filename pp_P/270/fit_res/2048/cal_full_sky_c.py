import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax = 1999
l = np.arange(lmax+1)

def cal_full_cl():
    cl_list = []
    for rlz_idx in range(1,100):
        print(f'{rlz_idx=}')
        if rlz_idx == 50:
            continue
        m = np.load(f'../../../../fitdata/2048/CMB/270/{rlz_idx}.npy')
        cl = hp.anafast(m, lmax=lmax)
        cl_list.append(cl)

    np.save('./full_sky_cl.npy', np.array(cl_list))


def check_full_cl():
    cl = np.load('./full_sky_cl.npy')
    print(f'{cl.shape=}')
    cl_B = cl[:,2,:]
    print(f'{cl_B.shape=}')
    bl = hp.gauss_beam(fwhm=np.deg2rad(9)/60, lmax=lmax, pol=True)[:,2]
    print(f'{bl.shape=}')

    dl_factor = l * (l+1) / (2*np.pi) / bl**2
    print(f'{dl_factor.shape=}')
    dl_B = cl_B * dl_factor
    print(f'{dl_B.shape=}')

    dl_B_mean = np.mean(dl_B, axis=0)
    dl_B_std = np.std(dl_B, axis=0)

    plt.plot(l, dl_B_mean, label='mean')
    plt.plot(l, dl_B_std, label='std')
    plt.semilogy()
    plt.legend()
    plt.show()

    # for rlz_idx in range(99):
    #     plt.plot(dl_B[rlz_idx,:])
    #     plt.semilogy()
    #     plt.show()

check_full_cl()

