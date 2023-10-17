import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('./cmbtqunoB20483.npy')
print(f'{m.shape}')
cl = np.load('./cmbcl3.npy')
print(f'{cl.shape}')
# mask = np.load('../../mask/north/BINMASKG2048.npy')
mask = np.load('../../ebleakage/circle_mask2048.npy')
fsky = np.sum(mask) / np.size(mask)
# apo_mask = np.load('../../mask/north/APOMASK2048C1_5.npy')
apo_mask = np.load('../../ebleakage/apo_circle_mask2048C1_8.npy')

lmax=1000
l = np.arange(lmax+1)
nside=2048

cut_cl = hp.anafast(m*mask, lmax=lmax)
cut_b = hp.alm2map(hp.map2alm(m*mask, lmax=lmax)[2], nside=nside)
cut_cl1 = hp.anafast(cut_b * mask, lmax=lmax)
cut_cl2 = hp.anafast(cut_b * apo_mask, lmax=lmax)
full_b = hp.alm2map(hp.map2alm(m, lmax=lmax)[2], nside=nside)
full_b_masked = hp.alm2map(hp.map2alm(m, lmax=lmax)[2], nside=nside) * mask
full_b_masked_cl = hp.anafast(full_b_masked, lmax=lmax)

hp.orthview(full_b, rot=[100,50,0], min=-0.6, max=0.6, half_sky=True, title='full sky B')
hp.orthview(cut_b, rot=[100,50,0], min=-0.6, max=0.6, half_sky=True, title='B from almB')
hp.orthview(cut_b * mask, rot=[100,50,0], min=-0.6, max=0.6, half_sky=True, title='cutB from almB')
hp.orthview(cut_b * apo_mask, rot=[100,50,0], min=-0.6, max=0.6, half_sky=True, title='apocutB from almB')
hp.orthview(full_b_masked, rot=[100,50,0], min=-0.6, max=0.6, half_sky=True, title='full b masked')
plt.show()



plt.loglog(l*(l+1)*cl[0:lmax+1,2]/(2*np.pi), label='theory')
plt.loglog(l*(l+1)*cut_cl[2]/(2*np.pi)/fsky, label='from cut QU')
plt.loglog(l*(l+1)*cut_cl1/(2*np.pi)/fsky, label='from cut B')
plt.loglog(l*(l+1)*cut_cl2/(2*np.pi)/fsky, label='from cut QU')
plt.loglog(l*(l+1)*full_b_masked_cl/(2*np.pi)/fsky, label='full b but have mask')
plt.legend()
plt.show()



