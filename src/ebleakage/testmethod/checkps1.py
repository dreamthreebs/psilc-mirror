import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax = 2000
l = np.arange(lmax+1)
def calc_dl(cl):
    return l*(l+1)*cl/(2*np.pi)

number = 0
crp_B = np.load(f'./fullqu_fitb/{number}/corrupted_B.npy')
cln_B = np.load(f'./fullqu_fitb/{number}/cleaned_B.npy')

cl_crp_B = hp.anafast(crp_B, lmax=lmax)
cl_cln_B = hp.anafast(cln_B, lmax=lmax)
dl_crp_B = calc_dl(cl_crp_B)
dl_cln_B = calc_dl(cl_cln_B)

plt.loglog(dl_crp_B, label='crp B fullqu_fitb')
plt.loglog(dl_cln_B, label='cln B fullqu_fitb')

crp_B = np.load(f'./zzr/{number}/corrupted_B.npy')
cln_B = np.load(f'./zzr/{number}/cleaned_B.npy')

cl_crp_B = hp.anafast(crp_B, lmax=lmax)
cl_cln_B = hp.anafast(cln_B, lmax=lmax)
dl_crp_B = calc_dl(cl_crp_B)
dl_cln_B = calc_dl(cl_cln_B)

plt.loglog(dl_crp_B, label='crp B direct on b')
plt.loglog(dl_cln_B, label='cln B direct on b')




plt.xlabel('$\\ell$')
plt.ylabel('$D_\\ell$')

plt.legend()
plt.show()






