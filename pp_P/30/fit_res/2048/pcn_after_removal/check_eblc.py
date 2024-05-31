import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax = 1999
nside = 2048
l = np.arange(lmax+1)

mask = np.load('../../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')

rmv_q = np.load('./3sigma/map_q_1.npy')
rmv_u = np.load('./3sigma/map_u_1.npy')
# hp.orthview(rmv_q, rot=[100,50,0])
# plt.show()


m = np.load('./3sigma/B/map_cln_b1.npy')
cn = np.load('../../../../../fitdata/synthesis_data/2048/CMBNOISE/30/1.npy')

alm_i, alm_e, alm_b = hp.map2alm([np.zeros_like(rmv_q), rmv_q*mask, rmv_u*mask], lmax=lmax)
no_eblc_b = hp.alm2map(alm_b, nside=nside)

alm_i, alm_e, alm_b = hp.map2alm(cn, lmax=lmax)
cn_b = hp.alm2map(alm_b, nside=nside)

alm_i, alm_e, alm_b = hp.map2alm(cn * mask, lmax=lmax)
masked_cn_b = hp.alm2map(alm_b, nside=nside)

# cl_b = hp.anafast(m, lmax=lmax)
# cl_cn_b = hp.anafast(cn_b, lmax=lmax)
# plt.semilogy(l*(l+1)*cl_b/(2*np.pi), label='rmv_b')
# plt.semilogy(l*(l+1)*cl_cn_b/(2*np.pi), label='cn_b')
# plt.legend()
# plt.show()

hp.orthview(cn_b*mask - no_eblc_b*mask, rot=[100,50,0], title='no_eblc_b', min=-2.5, max=2.5)
hp.orthview(cn_b*mask - m, rot=[100,50,0], title='after eblc', min=-2.5, max=2.5)
hp.orthview(no_eblc_b*mask - m, rot=[100,50,0], title='residual')

hp.orthview(cn_b*mask-masked_cn_b, rot=[100,50,0], title='lkg', min=-2.5, max=2.5)
plt.show()

