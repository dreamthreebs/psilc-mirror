import numpy as np
import matplotlib.pyplot as plt
import healpy as hp


def harmonic_ilc(std_maps, lmax, nside, wl=None):
    def maps2alms(sim, lmax):
        alms_list = []
        for index in range(len(sim)):
            _alm = hp.map2alm(sim[index], lmax=lmax)
            alms_list.append(_alm)
        alms = np.array(alms_list)
        return alms
    n_bands = len(std_maps)

    std_alms = maps2alms(std_maps, lmax=lmax)
    print(f'{std_alms.shape = }')

    if wl is None:
        R = np.empty((lmax+1, n_bands, n_bands))
        for i in range(n_bands):
            for j in range(n_bands):
                R[:, i, j] = hp.alm2cl(std_alms[i], std_alms[j])
        invR = np.linalg.pinv(R[2:])
        # invR = myinv(R[2:])
        oneVec = np.ones(n_bands)
        wl_2 = (oneVec@invR).T/(oneVec@invR@oneVec + 1e-12)
        wl = np.zeros((n_bands, lmax + 1))
        wl[:,2:] = wl_2

    ilc_alms_list = []
    for i in range(n_bands):
        ilc_alms_list.append(hp.almxfl(std_alms[i], wl[i], inplace=False))

    ilc_alm = np.sum(np.array(ilc_alms_list), axis=0)
    return wl, ilc_alm

if __name__ == '__main__':
    lmax = 500
    n_freq = 8
    nside = 512
    mask = np.load('../mask/north/APOMASKC1_10.npy')
    bin_mask = np.load('../mask/north/BINMASKG.npy')
    sim = np.load(f'../eblc/eblc_data/smcmbfg/data.npy') * mask
    fg = np.load(f'../eblc/eblc_data/smfg/data.npy') * mask
    # noise = np.load(f'../smooth/FULL_PATCH/noPS_northNOI/NOISE/B/data.npy')
    # bl = np.load('../smooth/BL/bl_std_curl.npy')
    print(f'sim.shape = {sim.shape}')

    wl, ilc_alm = harmonic_ilc(sim, lmax=lmax, nside=nside)
    ilc_map = hp.alm2map(ilc_alm, nside=nside) * bin_mask
    
    _, fg_res_alm = harmonic_ilc(fg, lmax=lmax, nside=nside, wl=wl)
    fgres_map = hp.alm2map(fg_res_alm, nside=nside) * bin_mask
    
    # _, noise_res_alm = harmonic_ilc(noise, wl=wl)
    ilc_cl = hp.alm2cl(ilc_alm, lmax=lmax)
    fg_res_cl = hp.alm2cl(fg_res_alm, lmax=lmax)
    # noise_res_cl = hp.alm2cl(noise_res_alm, lmax=lmax)
    
    # np.save('./hilcres/wl.npy',wl)
    # np.save('./hilcres/hilc_cl.npy',ilc_cl)
    # np.save('./hilcres/hilc_fgres_cl',fg_res_cl)
    # np.save('./hilcres/hilc_map.npy', ilc_map)
    # np.save('./hilcres/hilc_fgres_map.npy', fgres_map)
    
    # np.save('./hilcres/hilc_noise_cl',noise_res_cl)
    

