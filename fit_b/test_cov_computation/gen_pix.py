import numpy as np
import healpy as hp

from pathlib import Path
class GENPIX:
    def __init__(self, nside, lon, lat, radius_factor, beam, flux_idx):
        self.nside = nside
        self.lon = lon
        self.lat = lat
        self.radius_factor = radius_factor
        self.beam = beam
        self.flux_idx = flux_idx

    def gen_pix(self):
        ctr0_pix = hp.ang2pix(nside=self.nside, theta=self.lon, phi=self.lat, lonlat=True)
        self.ctr0_vec = np.array(hp.pix2vec(nside=self.nside, ipix=ctr0_pix)).astype(np.float64)
        self.ipix_fit = hp.query_disc(nside=self.nside, vec=self.ctr0_vec, radius=self.radius_factor * np.deg2rad(self.beam) / 60)
        path_pix_idx = Path(f'./pix_idx_qu')
        path_pix_idx.mkdir(exist_ok=True, parents=True)
        np.save(path_pix_idx / Path(f'{self.flux_idx}.npy'), self.ipix_fit)
        print(f'ndof = {len(self.ipix_fit)*2}')


obj = GENPIX(nside=2048, lon=0, lat=0, radius_factor=2.5, beam=67, flux_idx=0)
obj.gen_pix()

