import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from pathlib import Path

nside = 1024
npix = hp.nside2npix(nside)
beam = 11
rlz_idx = 0
lmax = 2500
l = np.arange(lmax+1)

ps = np.load('./data/ps_map.npy')
cmb = np.load(f'./data/cmb_i.npy')
noise = np.load(f'./data/noise_i.npy')

def check_cmb_cl():

    cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    cl_I = cls.T[0,:lmax+1]
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)
    cl_cmb = cl_I * bl**2

    cl_exp = hp.anafast(cmb, lmax=lmax)

    plt.loglog(l*(l+1)*cl_cmb, label='true')
    plt.loglog(l*(l+1)*cl_exp, label='exp')
    plt.show()

# check_cmb_cl()

def check_noise_cl():
    map_depth = 1.34
    cl_noise = (map_depth*np.ones(shape=(lmax+1,)))**2 / 3437.748**2

    cl_exp = hp.anafast(noise, lmax=lmax)

    plt.loglog(l*(l+1)*cl_noise, label='true')
    plt.loglog(l*(l+1)*cl_exp, label='exp')
    plt.show()

# check_noise_cl()

def tegmark_mf():

    pcn = np.load('./data/pcn.npy')
    cn = np.load('./data/cn.npy')
    cl_cn = hp.anafast(cn, lmax=lmax)

    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)

    Fl = bl / cl_cn * np.max(ps) / 3752502.1314773946
    print(f"{Fl[0:2]=}")
    Fl[0:2] = 0

    plt.loglog(Fl, label='Fl')
    plt.loglog(bl, label='bl')
    plt.loglog(cl_cn, label='cl_cn')
    plt.legend()
    plt.show()

    ps_out = hp.smoothing(ps, beam_window=Fl, pol=False)
    cn_out = hp.smoothing(cn, beam_window=Fl, pol=False)
    pcn_out = hp.smoothing(pcn, beam_window=Fl, pol=False)

    sigma = np.std(cn_out)
    snr = pcn_out / sigma

    print(f'{sigma=}')
    print(np.max(ps))
    print(np.max(ps_out))
    hp.mollview(ps_out, rot=[0,0,0], title='ps_out')
    hp.mollview(ps, rot=[0,0,0], title='ps')
    hp.mollview(cn_out, rot=[0,0,0], title='cn_out')
    hp.mollview(cn, rot=[0,0,0], title='cn')
    hp.mollview(pcn_out, rot=[0,0,0], title='pcn_out')
    hp.mollview(pcn, rot=[0,0,0], title='pcn')
    hp.mollview(snr, rot=[0,0,0], title='snr')

    # hp.gnomview(ps_out * np.max(ps) / np.max(ps_out) , rot=[0,0,0])
    plt.show()

tegmark_mf()


