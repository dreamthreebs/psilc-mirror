import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

from eblc_base import EBLeakageCorrection

freq = 155
df = pd.read_csv(f'../../../../../mask/mask_csv/{freq}.csv')
beam = 17
nside = 2048
lmax = 1999
fig_size = 100
npix = hp.nside2npix(nside=nside)
flux_idx = 1


# mask_list = np.load('../2sigma/mask_1.npy')
# print(f'{mask_list=}')

pcn = np.load(f'../../../../../../fitdata/synthesis_data/2048/PSCMBNOISE/{freq}/1.npy')
cn = np.load(f'../../../../../../fitdata/synthesis_data/2048/CMBNOISE/{freq}/1.npy')

lon = np.rad2deg(df.at[flux_idx, 'lon'])
lat = np.rad2deg(df.at[flux_idx, 'lat'])
ctr_vec = hp.ang2vec(theta=lon, phi=lat, lonlat=True)
pix_disc = hp.query_disc(nside=nside, vec=ctr_vec, radius=1.5 * np.deg2rad(beam)/60)

mask = np.ones(npix)
mask[pix_disc] = 0

pcn_i = pcn[0]
pcn_q = pcn[1]
pcn_u = pcn[2]
cn_i, cn_q, cn_u = cn

obj_eblc = EBLeakageCorrection(m=np.array([pcn_i, pcn_q*mask, pcn_u*mask]), lmax=lmax, nside=nside, mask=mask, post_mask=mask)
_, _, cln_b = obj_eblc.run_eblc()

hp.gnomview(pcn_q, rot=[lon, lat, 0], title='pcn_q', xsize=fig_size)
hp.gnomview(pcn_u, rot=[lon, lat, 0], title='pcn_u', xsize=fig_size)
hp.gnomview(cn_q, rot=[lon, lat, 0], title='cn_q', xsize=fig_size)
hp.gnomview(cn_u, rot=[lon, lat, 0], title='cn_u', xsize=fig_size)
plt.show()

alm_full_i, alm_full_e, alm_full_b = hp.map2alm([cn_i, cn_q, cn_u], lmax=lmax)
cn_full_e = hp.alm2map(alm_full_e, nside=nside)
cn_full_b = hp.alm2map(alm_full_b, nside=nside)

alm_full_i, alm_full_e, alm_full_b = hp.map2alm([pcn_i, pcn_q, pcn_u], lmax=lmax)
pcn_full_e = hp.alm2map(alm_full_e, nside=nside)
pcn_full_b = hp.alm2map(alm_full_b, nside=nside)

alm_i, alm_e, alm_b = hp.map2alm([pcn_i, pcn_q*mask, pcn_u*mask], lmax=lmax)
pcn_masked_e = hp.alm2map(alm_e, nside=nside)
pcn_masked_b = hp.alm2map(alm_b, nside=nside)

alm_i, alm_e, alm_b = hp.map2alm([cn_i, cn_q*mask, cn_u*mask], lmax=lmax)
cn_masked_e = hp.alm2map(alm_e, nside=nside)
cn_masked_b = hp.alm2map(alm_b, nside=nside)

hp.gnomview(cn_masked_e, rot=[lon, lat, 0], title='cn_e', xsize=fig_size, sub=261)
hp.gnomview(cn_masked_b, rot=[lon, lat, 0], title='cn_b', xsize=fig_size, sub=267)
hp.gnomview(pcn_masked_e, rot=[lon, lat, 0], title='pcn_e', xsize=fig_size, sub=262)
hp.gnomview(pcn_masked_b, rot=[lon, lat, 0], title='pcn_b', xsize=fig_size, sub=268)

# hp.gnomview(cn_masked_e - pcn_masked_e, rot=[lon, lat, 0], title='cn_e - pcn_e', xsize=fig_size, sub=263)
hp.gnomview(cn_masked_b - pcn_masked_b, rot=[lon, lat, 0], title='cn_b - pcn_b', xsize=fig_size, sub=263)
hp.gnomview(cln_b, rot=[lon, lat, 0], title='cln b', xsize=fig_size, sub=269, min=-3.6, max=3.46)

hp.gnomview(pcn_masked_e*mask, rot=[lon, lat, 0], title='masked e', xsize=fig_size, sub=264)
hp.gnomview(pcn_masked_b*mask, rot=[lon, lat, 0], title='masked b', xsize=fig_size, sub=(2,6,10))

hp.gnomview((pcn_masked_e-cn_full_e)*mask, rot=[lon, lat, 0], title='e - cn_full_e', xsize=fig_size, sub=265)
hp.gnomview((pcn_masked_b-cn_full_b)*mask, rot=[lon, lat, 0], title='b - cn_full_b', xsize=fig_size, sub=(2,6,11))

hp.gnomview((pcn_masked_e-pcn_full_e)*mask, rot=[lon, lat, 0], title='e - pcn_full_e', xsize=fig_size, sub=266)
hp.gnomview((pcn_masked_b-pcn_full_b)*mask, rot=[lon, lat, 0], title='b - pcn_full_b', xsize=fig_size, sub=(2,6,12))

plt.show()




