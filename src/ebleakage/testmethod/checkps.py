import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax = 4000
l = np.arange(lmax+1)
def calc_dl(cl):
    return l*(l+1)*cl/(2*np.pi)

number = 0
crp_B = np.load(f'./cutqu_fitqu/{number}/corrupted_B.npy')
cln_B = np.load(f'./cutqu_fitqu/{number}/cleaned_B.npy')

cl_crp_B = hp.anafast(crp_B, lmax=lmax)
cl_cln_B = hp.anafast(cln_B, lmax=lmax)
dl_crp_B = calc_dl(cl_crp_B)
dl_cln_B = calc_dl(cl_cln_B)

plt.loglog(dl_crp_B, label='crp B cutqu_fitqu')
plt.loglog(dl_cln_B, label='cln B cutqu_fitqu')


crp_B = np.load(f'./cutqu_fitb/{number}/corrupted_B.npy')
cln_B = np.load(f'./cutqu_fitb/{number}/cleaned_B.npy')

cl_crp_B = hp.anafast(crp_B, lmax=lmax)
cl_cln_B = hp.anafast(cln_B, lmax=lmax)
dl_crp_B = calc_dl(cl_crp_B)
dl_cln_B = calc_dl(cl_cln_B)

plt.loglog(dl_crp_B, label='crp B cutqu_fitb')
plt.loglog(dl_cln_B, label='cln B cutqu_fitb')

crp_B = np.load(f'./fullqu_fitqu/{number}/corrupted_B.npy')
cln_B = np.load(f'./fullqu_fitqu/{number}/cleaned_B.npy')

cl_crp_B = hp.anafast(crp_B, lmax=lmax)
cl_cln_B = hp.anafast(cln_B, lmax=lmax)
dl_crp_B = calc_dl(cl_crp_B)
dl_cln_B = calc_dl(cl_cln_B)

plt.loglog(dl_crp_B, label='crp B fullqu_fitqu')
plt.loglog(dl_cln_B, label='cln B fullqu_fitqu')

crp_B = np.load(f'./fullqu_fitb/{number}/corrupted_B.npy')
cln_B = np.load(f'./fullqu_fitb/{number}/cleaned_B.npy')

cl_crp_B = hp.anafast(crp_B, lmax=lmax)
cl_cln_B = hp.anafast(cln_B, lmax=lmax)
dl_crp_B = calc_dl(cl_crp_B)
dl_cln_B = calc_dl(cl_cln_B)

plt.loglog(dl_crp_B, label='crp B fullqu_fitb')
plt.loglog(dl_cln_B, label='cln B fullqu_fitb')
plt.xlabel('$\\ell$')
plt.ylabel('$D_\\ell$')

plt.legend()
plt.show()






