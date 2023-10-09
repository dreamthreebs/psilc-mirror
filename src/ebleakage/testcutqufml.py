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

full_cl = hp.anafast(m, lmax=lmax)

cut_alms = hp.map2alm(m * bin_mask, lmax=lmax)
cut_almT, cut_almE, cut_almB = [alm for alm in cut_alms]

fml_B = hp.alm2map([cut_almT, np.zeros_like(cut_almT), cut_almB], nside=nside)

masked_fml_B = fml_B * bin_mask

iter_1 = hp.alm2map([cut_almT, np.zeros_like(cut_almT), hp.map2alm(masked_fml_B, lmax=lmax)[2]], nside=nside) * bin_mask
iter_2 = hp.alm2map([cut_almT, np.zeros_like(cut_almT), hp.map2alm(iter_1, lmax=lmax)[2]], nside=nside) * bin_mask
iter_3 = hp.alm2map([cut_almT, np.zeros_like(cut_almT), hp.map2alm(iter_2, lmax=lmax)[2]], nside=nside) * bin_mask
iter_4 = hp.alm2map([cut_almT, np.zeros_like(cut_almT), hp.map2alm(iter_3, lmax=lmax)[2]], nside=nside) * bin_mask
# iter_4 = hp.alm2map([hp.map2alm(iter_3, lmax=300)[2], np.zeros_like(hp.map2alm(iter_3, lmax=300)[2]), hp.map2alm(iter_3, lmax=300)[2]], nside=nside) * bin_mask

# iter_5 = hp.alm2map([cut_almT, np.zeros_like(cut_almT), hp.map2alm(iter_4, lmax=lmax)[2]], nside=nside) * bin_mask
# iter_6 = hp.alm2map([cut_almT, np.zeros_like(cut_almT), hp.map2alm(iter_5, lmax=lmax)[2]], nside=nside) * bin_mask
# iter_7 = hp.alm2map([cut_almT, np.zeros_like(cut_almT), hp.map2alm(iter_6, lmax=lmax)[2]], nside=nside) * bin_mask
# iter_8 = hp.alm2map([cut_almT, np.zeros_like(cut_almT), hp.map2alm(iter_7, lmax=lmax)[2]], nside=nside) * bin_mask
# iter_9 = hp.alm2map([cut_almT, np.zeros_like(cut_almT), hp.map2alm(iter_8, lmax=lmax)[2]], nside=nside) * bin_mask

# iter_9 = hp.alm2map([hp.map2alm(iter_3, lmax=300)[2], np.zeros_like(hp.map2alm(iter_3, lmax=300)[2]), hp.map2alm(iter_8, lmax=300)[2]], nside=nside) * bin_mask

# iter_10 = hp.alm2map([cut_almT, np.zeros_like(cut_almT), hp.map2alm(iter_9, lmax=lmax)[2]], nside=nside) * bin_mask
# iter_11 = hp.alm2map([cut_almT, np.zeros_like(cut_almT), hp.map2alm(iter_10, lmax=lmax)[2]], nside=nside) * bin_mask
# iter_12 = hp.alm2map([cut_almT, np.zeros_like(cut_almT), hp.map2alm(iter_11, lmax=lmax)[2]], nside=nside) * bin_mask
# iter_13 = hp.alm2map([cut_almT, np.zeros_like(cut_almT), hp.map2alm(iter_12, lmax=lmax)[2]], nside=nside) * bin_mask
# iter_14 = hp.alm2map([cut_almT, np.zeros_like(cut_almT), hp.map2alm(iter_13, lmax=lmax)[2]], nside=nside) * bin_mask
# iter_15 = hp.alm2map([cut_almT, np.zeros_like(cut_almT), hp.map2alm(iter_14, lmax=lmax)[2]], nside=nside) * bin_mask
# iter_16 = hp.alm2map([cut_almT, np.zeros_like(cut_almT), hp.map2alm(iter_15, lmax=lmax)[2]], nside=nside) * bin_mask
# iter_17 = hp.alm2map([cut_almT, np.zeros_like(cut_almT), hp.map2alm(iter_16, lmax=lmax)[2]], nside=nside) * bin_mask
# iter_18 = hp.alm2map([cut_almT, np.zeros_like(cut_almT), hp.map2alm(iter_17, lmax=lmax)[2]], nside=nside) * bin_mask
# iter_19 = hp.alm2map([cut_almT, np.zeros_like(cut_almT), hp.map2alm(iter_18, lmax=lmax)[2]], nside=nside) * bin_mask



# full_QBUB = hp.alm2map([cut_almT, np.zeros_like(cut_almT), hp.map2alm(m, lmax=lmax)[2]], nside=nside) * bin_mask
# full_QBUB = hp.alm2map([hp.map2alm(m, lmax=300)[2], np.zeros_like(hp.map2alm(m, lmax=300)[2]), hp.map2alm(m, lmax=300)[2]], nside=nside) * bin_mask

hp.orthview(hp.ma(iter_4[1], badval=0), half_sky=True, title='iter_9', badcolor='white', sub=(1,3,1))
hp.orthview(hp.ma(full_QBUB[1], badval=0), half_sky=True, title='full sky QBUB', badcolor='white', sub=(1,3,2))
hp.orthview(hp.ma(full_QBUB[1]-iter_9[1], badval=0), half_sky=True, title='residual', badcolor='white', sub=(1,3,3))
plt.show()

cut_cl = hp.anafast(masked_fml_B, lmax=lmax)
iter_1_cl = hp.anafast(iter_1, lmax=lmax)

# iter_2_cl = hp.anafast(iter_2, lmax=lmax)
# iter_3_cl = hp.anafast(iter_3, lmax=lmax)
iter_4_cl = hp.anafast(iter_4, lmax=lmax)
# iter_5_cl = hp.anafast(iter_5, lmax=lmax)
# iter_6_cl = hp.anafast(iter_6, lmax=lmax)
# iter_7_cl = hp.anafast(iter_7, lmax=lmax)
# iter_8_cl = hp.anafast(iter_8, lmax=lmax)

# iter_9_cl = hp.anafast(iter_9, lmax=lmax)
# iter_13_cl = hp.anafast(iter_13, lmax=lmax)
# iter_19_cl = hp.anafast(iter_19, lmax=lmax)

# np.save('iter_1_cl', iter_1_cl)
# np.save('iter_4_cl', iter_4_cl)
# np.save('iter_9_cl', iter_9_cl)
# np.save('iter_13_cl', iter_13_cl)
# np.save('iter_19_cl', iter_19_cl)


plt.loglog(l*(l+1)*full_cl[2]/(2*np.pi))
plt.loglog(l*(l+1)*cut_cl[2]/(2*np.pi)/fsky, label='iter 0')
plt.loglog(l*(l+1)*iter_1_cl[2]/(2*np.pi)/fsky, label='iter 1' )

# plt.loglog(l*(l+1)*iter_2_cl[2]/(2*np.pi)/fsky, label='iter 2')
# plt.loglog(l*(l+1)*iter_3_cl[2]/(2*np.pi)/fsky, label='iter 3')
plt.loglog(l*(l+1)*iter_4_cl[2]/(2*np.pi)/fsky, label='iter 4')
# plt.loglog(l*(l+1)*iter_5_cl[2]/(2*np.pi)/fsky, label='iter 5')
# plt.loglog(l*(l+1)*iter_6_cl[2]/(2*np.pi)/fsky, label='iter 6')
# plt.loglog(l*(l+1)*iter_7_cl[2]/(2*np.pi)/fsky, label='iter 7')
# plt.loglog(l*(l+1)*iter_8_cl[2]/(2*np.pi)/fsky, label='iter 8')
# plt.loglog(l*(l+1)*iter_9_cl[2]/(2*np.pi)/fsky, label='iter 9')

# plt.legend()
# plt.xlabel('$\\ell$')
# plt.ylabel('$D_\\ell$')
# plt.show()



