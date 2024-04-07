import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nside = 2048
beam = 17

rlz_idx = 0

mask = hp.read_map('../mask/mask.fits', field=0)
ps_cmb_noise = hp.read_map(f'../input/{rlz_idx}.fits', field=0)
m = ps_cmb_noise * mask

# hp.mollview(ps, xsize=4000)
# hp.mollview(ps*mask, xsize=4000)
# hp.mollview(m, xsize=4000)
# plt.show()

src_idx = np.load('../source_indices.npy')
print(f'{src_idx=}')

for idx_src in np.arange(len(src_idx)):
    print(f'{idx_src=}')

    src_lon, src_lat = hp.pix2ang(nside=nside, ipix=src_idx[idx_src], lonlat=True)
    ipix_disc = hp.query_disc(nside=nside, vec=hp.pix2vec(nside=nside, ipix=src_idx[idx_src]), radius=1.5*np.deg2rad(beam)/60)

    hp.gnomview(m, rot=[src_lon, src_lat, 0], title='before inpaint')
    plt.show()
    
    
    neighbour_ipix = hp.get_all_neighbours(nside=nside, theta=ipix_disc)
    print(f'{neighbour_ipix.shape=}')
    
    for idx_itr in np.arange(2000):
        print(f'{idx_itr=}')
        for idx_disc in np.arange(neighbour_ipix.shape[1]):
            # print(f'{idx_disc=}')
            m[ipix_disc[idx_disc]] = np.mean(m[neighbour_ipix[:,idx_disc]])

    hp.gnomview(m, rot=[src_lon, src_lat, 0], title='after inpaint')
    plt.show()

np.save(f'./output/{rlz_idx}.npy',m)
# hp.mollview(m, xsize=4000)
# plt.show()



