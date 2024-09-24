import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from pathlib import Path

lmax = 600
nside = 2048
fg = np.load('../../fitdata/2048/FG/30/fg.npy')
apo_mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')

masked_fg = fg * apo_mask
hp.orthview(fg[0], rot=[100,50,0], half_sky=True)
hp.orthview(masked_fg[0], rot=[100,50,0], half_sky=True)
plt.show()

fsky = np.sum(apo_mask) / np.size(apo_mask)
alm_T, alm_E, alm_B = hp.map2alm(fg, lmax=3*nside-1)
m_B = hp.alm2map(alm_B, nside=nside)
m_E = hp.alm2map(alm_E, nside=nside)
m_T = hp.alm2map(alm_T, nside=nside)

cl_fg_TT = hp.anafast(m_T * apo_mask, lmax=3*nside-1)[0:lmax+1] / fsky
cl_fg_EE = hp.anafast(m_E * apo_mask, lmax=3*nside-1)[0:lmax+1] / fsky
cl_fg_BB = hp.anafast(m_B * apo_mask, lmax=3*nside-1)[0:lmax+1] / fsky

cl_fg = hp.anafast(masked_fg, lmax=lmax)
l = np.arange(lmax+1)
plt.loglog(l*(l+1)*cl_fg[0]/(2*np.pi) / fsky, label='msk on QU dl_fg T')
plt.loglog(l*(l+1)*cl_fg[1]/(2*np.pi) / fsky, label='msk on QU dl_fg E')
plt.loglog(l*(l+1)*cl_fg[2]/(2*np.pi) / fsky, label='msk on QU dl_fg B')
plt.loglog(l*(l+1)*cl_fg_TT/(2*np.pi), label='msk on T dl_fg T')
plt.loglog(l*(l+1)*cl_fg_EE/(2*np.pi), label='msk on E dl_fg E')
plt.loglog(l*(l+1)*cl_fg_BB/(2*np.pi), label='msk on B dl_fg B')

plt.legend()
plt.show()

cl_fg_TT[0:2] = 0
cl_fg_EE[0:2] = 0
cl_fg_BB[0:2] = 0

path_data = Path(f'./data_full_lmax3nside')
path_data.mkdir(exist_ok=True, parents=True)
np.save(path_data / Path('cl_fg_TT.npy'), cl_fg_TT)
np.save(path_data / Path('cl_fg_EE.npy'), cl_fg_EE)
np.save(path_data / Path('cl_fg_BB.npy'), cl_fg_BB)


