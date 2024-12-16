import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from pathlib import Path

def gen_ps(beam, nside, distance_factor=5, rlz_idx=0, flux_t1=5e2, flux_t2=3e2):

    lmax = 3*nside -1
    npix = hp.nside2npix(nside)

    m_delta = np.zeros((npix,))

    lon1 = 0
    lat1 = 0
    ipix_ctr1 = hp.ang2pix(theta=lon1, phi=lat1, lonlat=True, nside=nside)
    pix_lon1, pix_lat1 = hp.pix2ang(ipix=ipix_ctr1, nside=nside, lonlat=True)

    path_ps = Path('./data/ps')
    path_ps.mkdir(exist_ok=True, parents=True)

    print(f'{distance_factor=}')

    lon2 = distance_factor * beam / 60
    lat2 = 0
    ipix_ctr2 = hp.ang2pix(theta=lon2, phi=lat2, lonlat=True, nside=nside)
    pix_lon2, pix_lat2 = hp.pix2ang(ipix=ipix_ctr2, nside=nside, lonlat=True)

    m_delta[ipix_ctr1] = flux_t1
    m_delta[ipix_ctr2] = flux_t2
    sm_m = hp.smoothing(m_delta, fwhm=np.deg2rad(beam)/60)

    angdist = hp.rotator.angdist(dir1=(lon1,lat1), dir2=(lon2,lat2), lonlat=True)
    print(f'{np.rad2deg(angdist)=}')

    # np.save(path_ps / Path(f'ps_map_{distance_factor}.npy'), sm_m)

    nstd = 1
    np.random.seed(seed=rlz_idx)
    noise = nstd * np.random.normal(loc=0, scale=1, size=(npix,))
    m = sm_m + noise

    # np.save(path_ps / Path(f'ps_map_{distance_factor}.npy'), m)
    # np.save(path_ps / Path(f'pix_lon1.npy'), pix_lon1)
    # np.save(path_ps / Path(f'pix_lat1.npy'), pix_lat1)
    if rlz_idx == 3:
        np.save(path_ps / Path(f'pix_lon2_{distance_factor}.npy'), pix_lon2) # degree
        np.save(path_ps / Path(f'pix_lat2_{distance_factor}.npy'), pix_lat2) # degree

    hp.gnomview(m, rot=[pix_lon1, pix_lat1, 0])
    hp.gnomview(m, rot=[pix_lon2, pix_lat2, 0])
    plt.show()

    return m

if __name__=="__main__":
    gen_ps(beam=30, nside=1024, distance_factor=0.3, rlz_idx=0)



