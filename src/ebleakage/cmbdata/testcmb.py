import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

# cl = np.load('./cmbcl.npy')
# cl1 = np.load('./cmbcl1.npy')
# l = np.arange(len(cl))

# plt.loglog(l*(l+1)*cl[:,1]/(2*np.pi))
# plt.loglog(l*(l+1)*cl1[:,1]/(2*np.pi))
# plt.show()

lmax = 2000
nside = 2048
mask = np.load('../circle_mask2048.npy')

m = np.load('./cmbtqunoB2048.npy')
print(f'{m.shape}')
fsky = np.sum(mask)/np.size(mask)

full_clbb = hp.anafast(m, lmax=lmax)[2]

cut_alm = hp.map2alm(m * mask, lmax=lmax)
cutalmT, cutalmE, cutalmB = [x for x in cut_alm]

Bmap = hp.alm2map(cutalmB, nside=nside)

# hp.orthview(hp.ma(Bmap*mask, badval=0), half_sky=True, min=-0.3, max=0.3, cmap='jet', title='only operate on B')
hp.orthview(Bmap, half_sky=True, min=-0.3, max=0.3, title='from masked QU')
# plt.show()

fml_b = hp.alm2map([cutalmT, np.zeros_like(cutalmT), cutalmB], nside=nside) # (3,npix)

Bmap_fml = hp.alm2map(hp.map2alm(fml_b, lmax=lmax)[2], nside=nside)
hp.orthview(Bmap_fml, half_sky=True, min=-0.3, max=0.3, title='from full sky QU B fml')
hp.orthview(Bmap_fml-Bmap, half_sky=True, min=-0.3, max=0.3, title='difference')
plt.show()

# Bmap_from_family = hp.alm2map(hp.map2alm(Bfamily, lmax=lmax)[2], nside=nside) * mask

# Bfamilymask = hp.alm2map(hp.map2alm(Bfamily*mask, lmax=lmax)[2], nside=nside) * mask

# hp.orthview(hp.ma(BfamilyB, badval=0), half_sky=True, min=-0.3, max=0.3, cmap='jet',title='mask after QU2B')
# hp.orthview(hp.ma(Bfamilymask, badval=0), half_sky=True, min=-0.3, max=0.3, cmap='jet', title='mask before QU2B')

l = np.arange(lmax+1)
def calc_dl(m):
    return l*(l+1)*hp.anafast(m, lmax=lmax)/(2*np.pi)



# dl0 = calc_dl(m*mask)[2]
# plt.loglog(dl0)

# dl1 = calc_dl(Bmap)
# plt.loglog(dl1)


# dl2 = calc_dl(Bmap * mask)
# plt.loglog(dl2)

# dl3 = calc_dl(fml_b)[2]
# plt.loglog(dl3)



# dl2 = calc_dl(BfamilyB)
# plt.loglog(dl2)
# dl3 = calc_dl(Bfamilymask)
# plt.loglog(dl3)
# plt.show()




