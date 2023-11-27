import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

df = pd.read_csv('../../FGSim/FreqBand')
fold = 1.0

for i in range(len(df)):
    beam = df.at[i, 'beam']
    freq = df.at[i, 'freq']
    print(f'{freq=}, {beam=}')
    data = pd.read_csv(f'./sort_by_iflux/{freq}.csv')
    
    n_ps = len(data)
    frac_threshold = 0.005
    n_effps = int(frac_threshold * n_ps)
    print(f'{n_effps=}')
    lon = data.loc[:n_effps, 'lon'].to_numpy()
    lat = data.loc[:n_effps, 'lat'].to_numpy()
    iflux = data.loc[:n_effps, 'iflux'].to_numpy()
    
    # ps_m = hp.read_map('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/30GHz/strongradiops_map_30GHz.fits', field=0)
    # hp.gnomview(ps_m, rot=[np.rad2deg(lon), np.rad2deg(lat), 0] )
    # plt.show()
    
    nside = 2048
    npix = hp.nside2npix(nside)
    m = np.ones(npix)
    
    vec = hp.ang2vec(theta=np.rad2deg(lon), phi=np.rad2deg(lat), lonlat=True)
    print(f'{vec.shape=}')
    
    for i in range(vec.shape[0]):
        ipix = hp.query_disc(nside=nside, vec=vec[i,:], radius=fold * np.deg2rad(beam)/60)
        m[ipix] = 0
    
    # hp.gnomview(m, rot=[np.rad2deg(lon[0]), np.rad2deg(lat[0]), 0] )
    # hp.projscatter(theta=np.rad2deg(lon), phi=np.rad2deg(lat), lonlat=True )
    # plt.show()

    root = Path(f'./i_maskdot5/{fold}')
    root.mkdir(parents=True, exist_ok=True)
    hp.write_map(f'./i_maskdot5/{fold}/{freq}.fits', m, overwrite=True)

