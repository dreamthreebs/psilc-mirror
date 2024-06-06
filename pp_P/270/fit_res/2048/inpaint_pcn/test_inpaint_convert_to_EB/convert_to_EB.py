import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax = 1999
nside = 2048


cn_q = hp.read_map('./cn_Q.fits')
cn_u = hp.read_map('./cn_U.fits')

inp_q = hp.read_map('./inp_Q.fits')
inp_u = hp.read_map('./inp_U.fits')
I = np.zeros_like(cn_q)

ori_mask = np.load('../../../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')

def check_qu_map():

    hp.orthview(inp_q, rot=[100,50,0], half_sky=True, title='inpaint q')
    hp.orthview(inp_q - cn_q, rot=[100,50,0], half_sky=True, title='inpaint q - cn_q')
    hp.orthview(inp_u, rot=[100,50,0], half_sky=True, title='inpaint u')
    hp.orthview(inp_u - cn_u, rot=[100,50,0], half_sky=True, title='inpaint u - cn_u')
    plt.show()

def cvt_to_EB():
    alm_i, alm_e, alm_b = hp.map2alm([I, inp_q, inp_u], lmax=lmax)
    inp_E = hp.alm2map(alm_e, nside=nside)
    inp_B = hp.alm2map(alm_b, nside=nside)

    alm_i, alm_e, alm_b = hp.map2alm([I, cn_q * ori_mask, cn_u * ori_mask], lmax=lmax)
    masked_E = hp.alm2map(alm_e, nside=nside)
    masked_B = hp.alm2map(alm_b, nside=nside)

    alm_i, alm_e, alm_b = hp.map2alm([I, cn_q, cn_u], lmax=lmax)
    full_E = hp.alm2map(alm_e, nside=nside)
    full_B = hp.alm2map(alm_b, nside=nside)

    hp.orthview(inp_B * ori_mask, rot=[100, 50, 0], title='inp B')
    hp.orthview(masked_B * ori_mask, rot=[100, 50, 0], title='masked B')
    hp.orthview(full_B * ori_mask, rot=[100, 50, 0], title='full B')
    hp.orthview((inp_B - full_B) * ori_mask, rot=[100, 50, 0], title='inp B - full B')
    hp.orthview((masked_B - full_B) * ori_mask, rot=[100, 50, 0], title='masked_B - full B')
    hp.orthview((masked_B - inp_B) * ori_mask, rot=[100, 50, 0], title='masked_B - inp B')
    plt.show()

    hp.orthview(inp_E * ori_mask, rot=[100, 50, 0], title='inp E')
    hp.orthview(masked_E * ori_mask, rot=[100, 50, 0], title='masked E')
    hp.orthview(full_E * ori_mask, rot=[100, 50, 0], title='full E')
    hp.orthview((inp_E - full_E) * ori_mask, rot=[100, 50, 0], title='inp E - full E')
    hp.orthview((masked_E - full_E) * ori_mask, rot=[100, 50, 0], title='masked_E - full E')
    hp.orthview((masked_E - inp_E) * ori_mask, rot=[100, 50, 0], title='masked_E - inp E')
    plt.show()

cvt_to_EB()












