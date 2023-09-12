import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

lmax = 300
n_freq = 8
cl_type = 'B'
sim = np.load(f'../smooth/FULL_PATCH/noPS_northNOI/SIM/B/data.npy')
fg = np.load(f'../smooth/FULL_PATCH/noPS_northNOI/FG/B/data.npy')
noise = np.load(f'../smooth/FULL_PATCH/noPS_northNOI/NOISE/B/data.npy')
bl = np.load('../smooth/BL/bl_std_curl.npy')

print(f'sim.shape = {sim.shape}')

def maps2alms(sim):
    alms_list = []
    for index in range(len(sim)):
        _alm = hp.map2alm(sim[index], lmax=lmax)
        alms_list.append(_alm)
    alms = np.array(alms_list)
    return alms

def harmonic_ilc(std_maps, wl=None):

    n_bands = len(std_maps)

    std_alms = maps2alms(std_maps)
    print(f'{std_alms.shape = }')

    if wl is None:
        R = np.empty((lmax+1, n_bands, n_bands))
        for i in range(n_bands):
            for j in range(n_bands):
                R[:, i, j] = hp.alm2cl(std_alms[i], std_alms[j])
        invR = np.linalg.inv(R[2:])
        # invR = myinv(R[2:])
        oneVec = np.ones(n_bands)
        wl_2 = (oneVec@invR).T/(oneVec@invR@oneVec + 1e-15)
        wl = np.zeros((n_bands, lmax + 1))
        wl[:,2:] = wl_2

    ilc_alms_list = []
    for i in range(n_bands):
        ilc_alms_list.append(hp.almxfl(std_alms[i], wl[i], inplace=False))

    ilc_alm = np.sum(np.array(ilc_alms_list), axis=0)
    return wl, ilc_alm

wl, ilc_alm = harmonic_ilc(sim)

# _, fg_res_alm = harmonic_ilc(fg, wl=wl)
# _, noise_res_alm = harmonic_ilc(noise, wl=wl)
# ilc_cl = hp.alm2cl(ilc_alm, lmax=lmax)
# fg_res_cl = hp.alm2cl(fg_res_alm, lmax=lmax)
# noise_res_cl = hp.alm2cl(noise_res_alm, lmax=lmax)

# np.save('./HILCRESULT/wl2.npy',wl)
# np.save('./HILCRESULT/hilc_cl2.npy',ilc_cl)
# np.save('./HILCRESULT/hilc_fgres_cl2',fg_res_cl)
# np.save('./HILCRESULT/hilc_noise_cl2',noise_res_cl)

n_sim = 50
noise_cl_sum = 0
for i in range(n_sim):
    print(f'loop:{i}')
    noise = np.load(f'../smooth/FULL_PATCH/noPS_northNOI/NOISESIM/{i}/{cl_type}/data.npy')
    _, noise_res_alm = harmonic_ilc(noise, wl=wl)
    noise_res_cl = hp.alm2cl(noise_res_alm, lmax=lmax)
    noise_cl_sum = noise_cl_sum + noise_res_cl

noise_res_avg = noise_cl_sum / n_sim
np.save('./HILCRESULT/hilc_noise_avg2',noise_res_avg)












