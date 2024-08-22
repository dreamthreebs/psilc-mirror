import numpy as np
import healpy as hp
import pandas as pd
from healpy.newvisufunc import projview, newprojplot
import matplotlib.pyplot as plt

def see_mask():
    mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
    print(np.sum(mask)/np.size(mask))
    # hp.mollview(mask)
    hp.orthview(mask, rot=[100,50,0], half_sky=True)
    projview(mask, graticule=True, graticule_labels=True,projection_type='mollweide')
    # newprojplot(theta=np.radians(40), phi=np.radians(50), marker="o", color="r", markersize=10);
    # hp.projscatter(theta=100,phi=50, lonlat=True)
    # hp.graticule(local=True)

    plt.show()

def see_ps_map():
    df = pd.read_csv('./mask/30.csv')
    flux_idx = 0

    m_ps = np.load('./data/ps_67.npy')
    # hp.orthview(m_ps[1], rot=[100,50,0], half_sky=True)
    # hp.orthview(m_ps[2], rot=[100,50,0], half_sky=True)

    lon = np.rad2deg(df.at[flux_idx, "lon"])
    lat = np.rad2deg(df.at[flux_idx, "lat"])
    hp.gnomview(m_ps[1], rot=[lon,lat,0])
    hp.gnomview(m_ps[2], rot=[lon,lat,0])

    plt.show()

def see_input_B_map():
    df = pd.read_csv('./mask/30.csv')
    flux_idx = 0

    m_ps = hp.read_map('./inpainting/input/0.fits')
    # hp.orthview(m_ps[1], rot=[100,50,0], half_sky=True)
    # hp.orthview(m_ps[2], rot=[100,50,0], half_sky=True)

    lon = np.rad2deg(df.at[flux_idx, "lon"])
    lat = np.rad2deg(df.at[flux_idx, "lat"])
    hp.gnomview(m_ps, rot=[lon,lat,0])

    plt.show()


# see_ps_map()
see_input_B_map()
