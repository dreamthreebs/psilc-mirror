import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


m = np.load('./data.npy')
m_cmb = np.load('../../../FGSim/CMB5/215.npy')
lmax=350
nside=512
bl_220 = hp.gauss_beam(fwhm=np.deg2rad(11)/60, lmax=lmax, pol=True)[:,2]
bl = hp.gauss_beam(fwhm=np.deg2rad(67)/60, lmax=lmax, pol=True)[:,2]

mb = hp.alm2map(hp.almxfl(hp.map2alm(m_cmb, lmax=350)[2], bl/bl_220), nside=nside)
bin_mask = np.load('../../../src/mask/north/BINMASKG.npy')

print(f'{m.shape=}')
hp.orthview(mb * bin_mask, rot=[100,50,0], half_sky=True)
hp.orthview(m[4], rot=[100,50,0], half_sky=True)
plt.show()

diff_m = m[0]-mb*bin_mask
hp.orthview(diff_m, rot=[100,50,0], half_sky=True)
plt.show()

l = np.arange(lmax+1)
mask = np.load('../../../src/mask/north_smooth/APOMASKC1_5.npy')
fsky = np.sum(mask)/np.size(mask)
cl = hp.anafast(diff_m*mask, lmax=lmax)
plt.semilogy(l*(l+1)*(cl)/(2*np.pi)/fsky/bl**2)
plt.show()
