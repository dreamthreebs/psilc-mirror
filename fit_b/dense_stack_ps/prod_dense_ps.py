import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from pathlib import Path

nside = 2048
beam = 67

def prod_ps_lon_lat():

    ps_lon_lat_list = [] # in degree

    first_ps_lon_lat = (100,50)
    second_ps_lon_lat = (105,60)
    third_ps_lon_lat = (110,70)
    test_ps_lon_lat = (107,55)
    ps_lon_lat_list.append(first_ps_lon_lat)
    # ps_lon_lat_list.append(second_ps_lon_lat)
    # ps_lon_lat_list.append(third_ps_lon_lat)

    print(f'{ps_lon_lat_list=}')
    print(f'{np.array(ps_lon_lat_list).shape=}')
    mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
    hp.mollview(mask)

    # angle = np.rad2deg(hp.rotator.angdist(dir1=test_ps_lon_lat, dir2=np.array(ps_lon_lat_list).T, lonlat=True))
    # print(f'{len(np.array(ps_lon_lat_list))=}')
    # print(f'{angle=}')
    # angle = np.rad2deg(hp.rotator.angdist(dir1=test_ps_lon_lat, dir2=second_ps_lon_lat, lonlat=True))
    # print(f'{angle=}')

    # hp.projscatter(theta=first_ps_lon_lat[0], phi=second_ps_lon_lat[1], lonlat=True)
    # hp.projscatter(theta=second_ps_lon_lat[0], phi=second_ps_lon_lat[1], lonlat=True)
    # plt.show()

    for lon in np.linspace(100,100+360,300):
        for lat in np.linspace(30,90,300):
            print(f'{lon=}, {lat=}')
            pix_idx = hp.ang2pix(nside=nside, theta=lon, phi=lat, lonlat=True)

            if mask[pix_idx] == 0:
                print(f'this point is not inside mask')
                continue

            dir_1 = (lon, lat)
            dir_2 = np.asarray(ps_lon_lat_list).T

            angle = np.rad2deg(hp.rotator.angdist(dir1=dir_1, dir2=dir_2, lonlat=True))
            print(f'{angle=}')
            min_angle = np.min(angle)

            if min_angle < 3 * beam / 60:
                continue

            ps_lon_lat_list.append((lon, lat))
            print(f'{ps_lon_lat_list=}')

            hp.projscatter(theta=lon, phi=lat, lonlat=True)

    print(f'{ps_lon_lat_list=}')
    print(f'{np.array(ps_lon_lat_list).shape=}')
    path_data = Path(f'./data')
    path_data.mkdir(exist_ok=True, parents=True)
    np.save(path_data / Path('ps_lon_lat.npy'), np.array(ps_lon_lat_list))

    plt.show()

def prod_ps_map():
    npix = hp.nside2npix(nside=nside)
    Delta_m = np.zeros(shape=(3, npix))

    ps_lon_lat_arr = np.load('./data/ps_lon_lat.npy')
    n_ps = len(ps_lon_lat_arr)
    print(f'{ps_lon_lat_arr.shape=}')
    print(f'{n_ps=}')

    seed = 4242
    rng = np.random.default_rng(seed)

    P = 10000
    phi = rng.uniform(0, 2*np.pi, size=n_ps)

    Delta_Q = P * np.cos(phi)
    Delta_U = P * np.sin(phi)
    print(f'{Delta_Q=}')
    print(f'{Delta_U=}')
    print(f'{Delta_Q**2 + Delta_U**2=}')

    for ps_idx in range(n_ps):
        lon, lat = ps_lon_lat_arr[ps_idx]
        print(f'{lon=}, {lat=}')
        pix_idx = hp.ang2pix(nside=nside, theta=lon, phi=lat, lonlat=True)
        Delta_m[1, pix_idx] = Delta_Q[ps_idx]
        Delta_m[2, pix_idx] = Delta_U[ps_idx]

    m_ps = hp.smoothing(Delta_m, fwhm=np.deg2rad(beam)/60)
    np.save(f'./data/ps_{beam}.npy', m_ps)

prod_ps_map()






