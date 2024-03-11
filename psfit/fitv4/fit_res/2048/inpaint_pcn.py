import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

def gen_mask(ori_mask, nside, radius_factor, beam):
    for rlz_idx in range(100):
        print(f'{rlz_idx=}')
        mask_list = np.load(f'./ps_cmb_noise_residual/2sigma/mask{rlz_idx}.npy')
        print(f'{mask_list=}')

        # m = np.load('./ps_cmb_noise_residual/map0.npy')
        # m = np.load('../../../../fitdata/synthesis_data/2048/PSCMBNOISE/40/0.npy')[0]

        # hp.orthview(m*mask, rot=[100,50,0], half_sky=True)
        # plt.show()

        mask = np.copy(ori_mask)
        for flux_idx in mask_list:
            print(f'{flux_idx=}')
            pcn_fit_lon = np.load(f'./PSCMBNOISE/1.5/idx_{flux_idx}/fit_lon.npy')[rlz_idx]
            pcn_fit_lat = np.load(f'./PSCMBNOISE/1.5/idx_{flux_idx}/fit_lat.npy')[rlz_idx]

            ctr0_pix = hp.ang2pix(nside=nside, theta=pcn_fit_lon, phi=pcn_fit_lat, lonlat=True)
            ctr0_vec = np.array(hp.pix2vec(nside=nside, ipix=ctr0_pix)).astype(np.float64)

            ipix_mask = hp.query_disc(nside=nside, vec=ctr0_vec, radius=radius_factor * np.deg2rad(beam) / 60)
            mask[ipix_mask] = 0

        # hp.orthview(m*ori_mask, rot=[100,50,0], half_sky=True)
        # hp.orthview(m*mask, rot=[100,50,0], half_sky=True)
        # plt.show()

        hp.write_map(f'./for_inpainting/mask/pcn/2sigma/{rlz_idx}.fits', mask, overwrite=True)


if __name__ == "__main__":
    mask = np.load('../../../../src/mask/north/BINMASKG2048.npy')
    nside = 2048
    radius_factor = 1.5
    beam = 63
    gen_mask(mask, nside=nside, radius_factor=radius_factor, beam=beam)

