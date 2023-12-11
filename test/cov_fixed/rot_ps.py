import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd

beam = 63 # arcmin
sigma = np.deg2rad(beam)/60 / (np.sqrt(8*np.log(2)))
print(f'{sigma=}')

nside = 2048

# m = np.load('../../FGSim/STRPSCMBFGNOISE/40.npy')[0]
# m = np.load('../../FGSim/STRPSFGNOISE/40.npy')[0]
# m = np.load('../../FGSim/STRPSCMBNOISE/40.npy')[0]
m = np.load('../../FGSim/PSNOISE/2048/40.npy')[0]
# nstd = np.load('../../FGSim/NSTDNORTH/2048/40.npy')[0]
# cstd = np.ones(hp.nside2npix(nside)) *  75.2896

df = pd.read_csv('../ps_sort/sort_by_iflux/40.csv')
lon = df.at[44, 'lon']
lat = df.at[44, 'lat']
iflux = df.at[44, 'iflux']

print(f'{iflux=}')

def see_true_map():
    hp.gnomview(m, rot=[np.rad2deg(lon), np.rad2deg(lat), 0])
    plt.show()
    
    vec = hp.ang2vec(theta=np.rad2deg(lon), phi=np.rad2deg(lat), lonlat=True)
    
    ipix_disc = hp.query_disc(nside=nside, vec=vec, radius=np.deg2rad(beam)/60)
    
    mask = np.ones(hp.nside2npix(nside))
    mask[ipix_disc] = 0
    
    hp.gnomview(mask, rot=[np.rad2deg(lon), np.rad2deg(lat), 0])
    plt.show()

# see_true_map()

psi = 0
theta = np.pi / 2
phi = np.pi/2
print(f'{psi=},{theta=},{phi=}')

r = hp.Rotator(rot=[90, 90, 0])
o_theta, o_phi = np.pi/2, 0
r_theta, r_phi = r(o_theta, o_phi)
print(f'{r_theta=}, {r_phi=}')


