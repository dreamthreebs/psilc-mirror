import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

freq = 270
df = pd.read_csv(f'../../../../mask/mask_csv/{freq}.csv')
flux_idx = 1
lon = np.rad2deg(df.at[flux_idx, 'lon'])
lat = np.rad2deg(df.at[flux_idx, 'lat'])

def check_Q():

    Q = hp.read_map('./3sigma/input/Q/1.fits')
    
    inp_Q = hp.read_map('./3sigma/QU/Q_output/1.fits')
    mask = np.load('../../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
    
    # hp.orthview(Q*mask, rot=[100,50,0], title='pcn Q', half_sky=True)
    # hp.orthview(inp_Q*mask, rot=[100,50,0], title='inpainted Q', half_sky=True)
    # plt.show()
    
    hp.gnomview(Q, rot=[lon, lat, 0], title='pcn Q', xsize=60)
    hp.gnomview(inp_Q, rot=[lon, lat, 0], title='inp Q', xsize=60)
    plt.show()

def check_U():

    U = hp.read_map('./3sigma/input/U/1.fits')
    
    inp_U = hp.read_map('./3sigma/QU/U_output/1.fits')
    mask = np.load('../../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
    
    # hp.orthview(Q*mask, rot=[100,50,0], title='pcn Q', half_sky=True)
    # hp.orthview(inp_Q*mask, rot=[100,50,0], title='inpainted Q', half_sky=True)
    # plt.show()
    
    hp.gnomview(U, rot=[lon, lat, 0], title='pcn U', xsize=60)
    hp.gnomview(inp_U, rot=[lon, lat, 0], title='inp U', xsize=60)
    plt.show()

check_U()
