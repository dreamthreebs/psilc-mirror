import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

nside = 2048
mask = np.load('../../src/mask/north/BINMASKG2048.npy')
df = pd.read_csv('./ps_in_mask/mask40.csv')

def check_pos_ps():
    hp.orthview(mask, rot=[100,50,0], half_sky=True)
    
    for i in range(len(df)):
        lon = df.at[i, 'lon']
        lat = df.at[i, 'lat']
        hp.projscatter(theta=np.rad2deg(lon), phi=np.rad2deg(lat), lonlat=True, s=1.2)
    plt.show()

def see_one_beam_size_in_mask():

    hp.orthview(mask, rot=[100,50,0], half_sky=True)
    lon = df.at[0,'lon']
    lat = df.at[0,'lat']
    hp.projscatter(theta=np.rad2deg(lon), phi=np.rad2deg(lat), lonlat=True, linewidth=1.0)
    plt.show()

    m = np.zeros(hp.nside2npix(nside=nside))
    ipix = hp.ang2pix(nside=nside, theta=np.rad2deg(lon), phi=np.rad2deg(lat), lonlat=True)
    m[ipix] = 1e6
    sm_m = hp.smoothing(m, fwhm=np.deg2rad(63)/60)

    hp.orthview(sm_m * mask, rot=[100,50,0], half_sky=True)
    plt.show()
def see_one_point():

    m = np.load('../../FGSim/PSNOISE/2048/40.npy')[0]
    m1 = hp.read_map('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/40GHz/strongradiops_map_40GHz.fits', field=0)
    # m1[m1>0] = 1

    lon = df.at[0,'lon']
    lat = df.at[0,'lat']
    hp.gnomview(mask, rot=[np.rad2deg(lon),np.rad2deg(lat),0], reso=1, xsize=400, ysize=400)
    hp.projscatter(theta=np.rad2deg(lon), phi=np.rad2deg(lat), lonlat=True, linewidth=1.0)
    hp.gnomview(m1, rot=[np.rad2deg(lon),np.rad2deg(lat),0], reso=1, xsize=400, ysize=400)
    # hp.gnomview(m, rot=[np.rad2deg(lon),np.rad2deg(lat),0])
    hp.projscatter(theta=np.rad2deg(lon), phi=np.rad2deg(lat), lonlat=True, linewidth=1.0)
    plt.show()


if __name__ == '__main__':
    # check_pos_ps()
    # see_one_beam_size_in_mask()
    see_one_point()
