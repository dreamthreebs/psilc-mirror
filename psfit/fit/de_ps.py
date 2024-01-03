import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

nside = 2048
beam = 63
sigma = np.deg2rad(beam) / 60 / (np.sqrt(8 * np.log(2)))

m = np.load('../../FGSim/PSNOISE/2048/40.npy')[0]
mask = np.load('../../src/mask/north/BINMASKG2048.npy')

mask_m = m * mask

hp.orthview(mask_m, rot=[100,50,0], half_sky=True, min=-30, max=30, title='ps noise map')
# plt.show()

def single_ps_model(norm_beam, sigma, theta):
    return norm_beam / (2 * np.pi * sigma**2) * np.exp(- (theta)**2 / (2 * sigma**2))


def gen_fit_ps_map():
    df = pd.read_csv('./data/40.csv')
    norm_beam_all = df['fit_norm']
    n_ps = np.count_nonzero(norm_beam_all)
    print(f'{n_ps=}')
    for i in range(n_ps):
        norm_beam = df.at[i, "fit_norm"]
        fit_lon = np.rad2deg(df.at[i, "fit_lon"])
        fit_lat = np.rad2deg(df.at[i, "fit_lat"])
        ctr_ipix = hp.ang2pix(nside=nside, theta=fit_lon, phi=fit_lat, lonlat=True)
        ctr_vec = hp.pix2vec(nside=nside, ipix=ctr_ipix)
        ipix_disc = hp.query_disc(nside=nside, vec=ctr_vec, radius=np.deg2rad(beam)/60)
        vec_around = np.array(hp.pix2vec(nside=nside, ipix=ipix_disc))
        theta = hp.rotator.angdist(dir1=ctr_vec, dir2=vec_around)

        fit_m = single_ps_model(norm_beam, sigma, theta)
        m[ipix_disc] = m[ipix_disc] - fit_m
        print(f'{fit_m.shape=}')

    return m


de_ps_m = gen_fit_ps_map()

hp.orthview(de_ps_m * mask, rot=[100,50,0], half_sky=True, title='de_ps', min=-30, max=30)
plt.show()


