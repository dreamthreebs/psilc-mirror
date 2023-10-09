import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


lmax = 2000
l = np.arange(lmax+1)
nside = 2048


m = np.load('./cmbdata/cmbtqunoB20483.npy')
bin_mask = np.load('./circle_mask2048.npy')
apo_mask = np.load('./apo_circle_mask2048C1_12.npy')
fsky = np.sum(bin_mask)/np.size(bin_mask)
fsky_apo = np.sum(apo_mask)/np.size(apo_mask)
print(f'{m.shape}')

# full_cl = hp.anafast(m, lmax=lmax)
cut_cl = hp.anafast(m * bin_mask, lmax=lmax)
alm = hp.map2alm(m, lmax=lmax)
fml = hp.alm2map([alm[0], np.zeros_like(alm[0]), hp.map2alm(m*bin_mask, lmax=lmax)[2]], nside=nside) * bin_mask
fml_cl = hp.anafast(fml * bin_mask, lmax=lmax)


# iter_1_cl = np.load('./iter_1_cl.npy')
# iter_4_cl = np.load('./iter_4_cl.npy')
# iter_9_cl = np.load('./iter_9_cl.npy')
# iter_13_cl = np.load('./iter_13_cl.npy')
# iter_19_cl = np.load('./iter_19_cl.npy')

plt.loglog(l*(l+1)*cut_cl[2]/(2*np.pi)/fsky, label='cut sky cl' )
plt.loglog(l*(l+1)*fml_cl[2]/(2*np.pi)/fsky, label='fml sky cl' )
# plt.loglog(l*(l+1)*iter_1_cl[2]/(2*np.pi)/fsky, label='iter 1' )
# plt.loglog(l*(l+1)*iter_4_cl[2]/(2*np.pi)/fsky, label='iter 4' )
# plt.loglog(l*(l+1)*iter_9_cl[2]/(2*np.pi)/fsky, label='iter 9' )
# plt.loglog(l*(l+1)*iter_13_cl[2]/(2*np.pi)/fsky, label='iter 13' )
# plt.loglog(l*(l+1)*iter_19_cl[2]/(2*np.pi)/fsky, label='iter 19' )

plt.legend()
plt.xlabel('$\\ell$')
plt.ylabel('$D_\\ell$')
plt.show()



