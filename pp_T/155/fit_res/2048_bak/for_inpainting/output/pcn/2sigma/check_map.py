import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = hp.read_map('./0.fits',field=0)
m_cmb = hp.read_map('../../../input/pcn/2sigma/0.fits')
mask = np.load('../../../../../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
mask_ps = hp.read_map('../../../mask/pcn/2sigma/0.fits', field=0)
m_ps = np.load('../../../../../../../../fitdata/synthesis_data/2048_bak/PSCMBNOISE/155/0.npy')[0]
m_no_ps = np.load('../../../../../../../../fitdata/synthesis_data/2048_bak/CMBNOISE/155_bak/0.npy')[0]

hp.orthview(m*mask, rot=[100,50,0], half_sky=True, title='inpaint')
hp.orthview(m_cmb*mask, rot=[100,50,0], half_sky=True, title='origin')
hp.orthview(m_ps*mask, rot=[100,50,0], half_sky=True, title='numpy origin')
hp.orthview(mask_ps, rot=[100,50,0], half_sky=True, title='mask ps')
hp.orthview(m_no_ps*mask, rot=[100,50,0], half_sky=True, title='cmb+noise')
hp.orthview((m_ps - m_no_ps)*mask, rot=[100,50,0], half_sky=True, title='only ps')
hp.orthview((m-m_no_ps)*mask, rot=[100,50,0], half_sky=True, title='inpaint residual')
plt.show()
