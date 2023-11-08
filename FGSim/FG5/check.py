import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax=500
nside=512
m = np.load('./40.npy')
m_noPS = np.load('../FG_noPS5/40.npy')

# hp.mollview(m[0], title='I', norm='hist')
# hp.mollview(m[1], title='Q', norm='hist')
# hp.mollview(m[2], title='U', norm='hist')
# plt.show()

hp.mollview(np.abs(m[0]-m_noPS[0])+0.001, title='I', norm='log')
hp.mollview(np.abs(m[1]-m_noPS[1])+0.001, title='Q', norm='log')
hp.mollview(np.abs(m[2]-m_noPS[2])+0.001, title='U', norm='log')
plt.show()



m_teb = hp.alm2map(hp.map2alm(m-m_noPS, lmax=lmax), nside=nside)

hp.mollview(m_teb[0], title='T', norm='hist')
hp.mollview(m_teb[1], title='E', norm='hist')
hp.mollview(m_teb[2], title='B', norm='hist')
plt.show()


