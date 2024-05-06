import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

freq = 270
lmax = 1999
beam = 9
l = np.arange(lmax+1)
bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)

bin_mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
apo_mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')

m_q = np.load('./pcn_after_removal/2sigma/map_q_1.npy') * bin_mask * apo_mask
m_u = np.load('./pcn_after_removal/2sigma/map_u_1.npy') * bin_mask * apo_mask

pcn = np.load(f'../../../../fitdata/synthesis_data/2048/PSCMBNOISE/{freq}/1.npy') * apo_mask
pcn_q = pcn[1].copy()
pcn_u = pcn[2].copy()

cn = np.load(f'../../../../fitdata/synthesis_data/2048/CMBNOISE/{freq}/1.npy') * apo_mask
cn_q = cn[1].copy()
cn_u = cn[2].copy()

dl_factor = l * (l + 1) / (2 * np.pi) / bl**2
dl_pcn = dl_factor * hp.anafast([np.zeros_like(pcn_q), pcn_q, pcn_u], lmax=lmax)
dl_cn = dl_factor * hp.anafast([np.zeros_like(cn_q), cn_q, cn_u], lmax=lmax)
dl_removal = dl_factor * hp.anafast([np.zeros_like(cn_q), m_q, m_u], lmax=lmax)

plt.figure(1)
plt.plot(l, dl_pcn[1], label='pcn EE')
plt.plot(l, dl_cn[1], label='cn EE')
plt.plot(l, dl_removal[1], label='removal EE')

plt.figure(2)
plt.plot(l, dl_pcn[2], label='pcn BB')
plt.plot(l, dl_cn[2], label='cn BB')
plt.plot(l, dl_removal[2], label='removal BB')


plt.show()


