import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

nside = 512
beam = 63

m = np.load('../../FGSim/PSNOISE/40.npy')[0]
nstd = np.load('../../FGSim/NSTDNORTH/40.npy')[0]
sigma_true = np.deg2rad(beam)/60 / (np.sqrt(8*np.log(2)))

ps_set = set()
for idx in np.arange(20000,30000):
    if idx % 1000 == 0:
        print(f'{idx=}')
    idx_pix = idx
    ct_theta, ct_phi = hp.pix2ang(nside=nside, ipix=idx_pix, lonlat=True)
    ct_vec = hp.pix2vec(nside=nside, ipix=idx_pix)
    
    # hp.gnomview(m, rot=[ct_theta, ct_phi, 0])
    # plt.show()
    
    ipix_ori_disc = hp.query_disc(nside=nside, vec=ct_vec, radius=np.deg2rad(beam)/60)
    data_ct = m[ipix_ori_disc]
    # print(f'{data_ct=}')
    
    idx_pix_top = ipix_ori_disc[np.where(data_ct==np.max(data_ct))[0][0]] # find nearby top
    
    # top_theta, top_phi = hp.pix2ang(nside=nside, ipix=idx_pix_top, lonlat=True) # center=26106351
    # hp.gnomview(m, rot=[top_theta, top_phi, 0])
    # plt.show()
    
    top_vec = np.array(hp.pix2vec(nside=nside, ipix=idx_pix_top)).astype(np.float64)
    # print(f'{top_vec=}')
    
    ipix_fit = hp.query_disc(nside=nside, vec=top_vec, radius=2 * np.deg2rad(beam)/60)
    # print(f'{ipix_fit.shape=}')
    
    def fit_model(theta, norm_beam, sigma, const):
        beam_profile = norm_beam / (2*np.pi*sigma**2) * np.exp(- (theta)**2 / (2 * sigma**2))
        # print(f'{beam_profile=}')
        return beam_profile + const
    
    vec_around = hp.pix2vec(nside=nside, ipix=ipix_fit.astype(int))
    # print(f'{vec_around=}')
    theta = np.arccos(np.array(top_vec) @ np.array(vec_around))
    theta = np.nan_to_num(theta)
    # theta = np.sqrt(2*(1-np.array(center_vec) @ np.array(vec_around)))
    
    # print(f'{theta=}')
    
    y_arr = m[ipix_fit]
    # print(f'{y_arr=}')
    y_err = nstd[ipix_fit]
    # print(f'{y_err=}')

    # plt.plot(ipix_fit, y_arr)
    # plt.show()
    
    lsq = LeastSquares(x=theta, y=y_arr, yerror=4*y_err, model=fit_model)
    
    obj_minuit = Minuit(lsq, norm_beam=1, sigma=0.0070, const=0)
    obj_minuit.limits = [(0,10),(1e-6,1e-2),(-100,100)]
    # print(obj_minuit.scan(ncall=100))
    # obj_minuit.errors = (0.1, 0.2)
    # print(obj_minuit.migrad())
    # print(obj_minuit.hesse())
    # print(obj_minuit.values)
    # print(obj_minuit.errors)

    obj_minuit.migrad()
    obj_minuit.hesse()

    threshold = 0.01
    if (np.abs(obj_minuit.values['sigma'] - sigma_true)/sigma_true < threshold) and (obj_minuit.values['norm_beam']>0.02):
        print(f'there is a point source near pixel:{idx_pix}, its center is at pixel {idx_pix_top}')
        print(obj_minuit.values)
        print(obj_minuit.errors)
        ps_set.add(idx_pix)
    else:
        # print(f'no point source near {idx_pix=}')
        pass

print(f'{ps_set=}')
print(f'{len(ps_set)}')



