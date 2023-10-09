import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

crp_QU = np.load('./QU/0/corrupted_QU.npy')
cln_QU = np.load('./QU/0/cleaned_QU.npy')
lmax = 2000
l = np.arange(lmax+1)
def calc_dl(cl):
    return l*(l+1)*cl/(2*np.pi)


cl_crp_QU = hp.anafast(crp_QU, lmax=lmax)[2]
cl_cln_QU = hp.anafast(cln_QU, lmax=lmax)[2]
dl_crp_QU = calc_dl(cl_crp_QU)
dl_cln_QU = calc_dl(cl_cln_QU)

crp_B = np.load('./B/0/corrupted_B.npy')
cln_B = np.load('./B/0/cleaned_B.npy')

cl_crp_B = hp.anafast(crp_B, lmax=lmax)
cl_cln_B = hp.anafast(cln_B, lmax=lmax)
dl_crp_B = calc_dl(cl_crp_B)
dl_cln_B = calc_dl(cl_cln_B)

plt.loglog(dl_crp_QU, label='crp QU')
plt.loglog(dl_cln_QU, label='cln QU')
plt.loglog(dl_crp_B, label='crp B')
plt.loglog(dl_cln_B, label='cln B')

number = 1

crp_QU = np.load(f'./QU/{number}/corrupted_QU.npy')
cln_QU = np.load(f'./QU/{number}/cleaned_QU.npy')
cl_crp_QU = hp.anafast(crp_QU, lmax=lmax)[2]
cl_cln_QU = hp.anafast(cln_QU, lmax=lmax)[2]
dl_crp_QU = calc_dl(cl_crp_QU)
dl_cln_QU = calc_dl(cl_cln_QU)

crp_B = np.load(f'./B/{number}/corrupted_B.npy')
cln_B = np.load(f'./B/{number}/cleaned_B.npy')

cl_crp_B = hp.anafast(crp_B, lmax=lmax)
cl_cln_B = hp.anafast(cln_B, lmax=lmax)
dl_crp_B = calc_dl(cl_crp_B)
dl_cln_B = calc_dl(cl_cln_B)

plt.loglog(dl_crp_QU, label='crp QU1')
plt.loglog(dl_cln_QU, label='cln QU1')
plt.loglog(dl_crp_B, label='crp B1')
plt.loglog(dl_cln_B, label='cln B1')




plt.legend()
plt.show()






