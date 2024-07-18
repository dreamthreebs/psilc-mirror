import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nside = 128
npix = hp.nside2npix(nside)
beam = 11

def test_noise_simulations():
    np.random.seed(seed=0)
    xiaosan = np.random.normal(loc=1, scale=33, size=(npix,))
    m = np.random.normal(loc=0, scale=1, size=(npix,))
    return m

def check_seed_noise():
    m0 = np.load('./test_seed_0.npy')
    m1 = np.load('./test_seed_2.npy')
    hp.mollview(m0)
    hp.mollview(m1)
    hp.mollview(m1-m0)
    plt.show()

def test_cmb_simulations():
    cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    np.random.seed(seed=0)
    # cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=1999)
    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=3*nside-1)
    return cmb_iqu

def check_seed_cmb():
    m0 = np.load('./test_seed_cmb_0.npy')[0]
    m1 = np.load('./test_seed_cmb_1.npy')[0]
    hp.mollview(m0)
    hp.mollview(m1)
    hp.mollview(m1-m0)
    plt.show()




# m = test_noise_simulations()
# np.save('./test_seed_2.npy', m)

# m = test_cmb_simulations()
# np.save('./test_seed_cmb_1.npy', m)

# check_seed_noise()
check_seed_cmb()


