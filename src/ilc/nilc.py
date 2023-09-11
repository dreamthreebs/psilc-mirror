import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pandas as pd

lmax=300
nside=512
cl_type = 'B'

class NILC:
    def __init__(self, needlet_config, weights_name=None, weights_config=None, Sm_alms=None, Sm_maps=None, mask=None, lmax=1000, nside=1024, Rtol=1/1000):

        self.weights_config = weights_config
        self.weights_name = weights_name
        self.Rtol = Rtol
        self.lmax = lmax
        if Sm_maps is not None:
            self.nmaps = Sm_maps.shape[0]
            self.npix = Sm_maps.shape[-1]
            self.nside = hp.npix2nside(self.npix)
            self.maps = Sm_maps
            if mask is not None:
                self.maps = Sm_maps * mask
            self.mask = mask

            Sm_alms_list = []
            for i in range(self.nmaps):
                Sm_alm = hp.map2alm(self.maps[i], lmax=lmax)
                Sm_alms_list.append(Sm_alm)
            self.alms = np.array(Sm_alms_list)

        if Sm_alms is not None:
            self.alms = Sm_alms
            self.nmaps = Sm_alms.shape[0]
            self.npix = hp.nside2npix(nside)
            self.nside = nside

        self.needlet = pd.read_csv(needlet_config)
        self.n_needlet = len(self.needlet)
        self.ilc_lmax = 800

    def calc_hl_bak(self):
        hl = np.zeros((self.n_needlet, self.lmax+1))
        for i in range(self.n_needlet):
            nlmax = self.needlet.at[i,'lmax']
            nlmin = self.needlet.at[i,'lmin']
            nlpeak = self.needlet.at[i,'lpeak']
            def funhl(l):
                if l < nlmin or l > nlmax:
                    return 0
                elif l < nlpeak:
                    return np.cos(((nlpeak-l)/(nlpeak-nlmin)) * np.pi/2)
                elif l > nlpeak:
                    return np.cos(((l-nlpeak)/(nlmax-nlpeak)) * np.pi/2)
                else:
                    return 1
            vecHl = np.vectorize(funhl, otypes=[float])
            hl[i] = vecHl(np.arange(self.lmax+1))
        self.hl1 = hl
        print(f'{self.hl.shape = }')
    def calc_hl(self):
        hl = np.zeros((self.n_needlet, self.lmax+1))
        l_range = np.arange(self.lmax+1)
        for i in range(self.n_needlet):
            nlmax = self.needlet.at[i,'lmax']
            nlmin = self.needlet.at[i,'lmin']
            nlpeak = self.needlet.at[i,'lpeak']

            condition1 = (l_range < nlmin) | (l_range > nlmax)
            condition2 = l_range < nlpeak
            condition3 = l_range > nlpeak
            eps=1e-15
            hl[i] = np.where(condition1, 0,
                       np.where(condition2, np.cos(((nlpeak-l_range)/(nlpeak-nlmin+eps)) * np.pi/2),
                       np.where(condition3, np.cos(((l_range-nlpeak)/(nlmax-nlpeak+eps)) * np.pi/2), 1)))
        self.hl = hl
    def calc_hl1(self):
        hl = np.zeros((self.n_needlet, self.lmax+1))
        l_range = np.arange(self.lmax+1)
        for i in range(self.n_needlet):
            nlmax = self.needlet.at[i,'lmax']
            nlmin = self.needlet.at[i,'lmin']
            nlpeak = self.needlet.at[i,'lpeak']
            hl[i,nlmin:nlmax] = 1
        self.hl = hl
    def calc_beta(self):
        hl = self.hl
        beta_list = []
        for j in range(self.n_needlet):
            beta_nside = self.needlet.at[j, 'nside']
            beta_npix = hp.nside2npix(beta_nside)
            beta = np.zeros((self.nmaps, beta_npix))
            for i in range(self.nmaps):
                beta_alm_ori = hp.almxfl(self.alms[i], self.hl[j])
                beta[i] = hp.alm2map(beta_alm_ori, beta_nside)

            beta_list.append(beta)
            print(f'{beta.shape = }')
        self.beta_list = beta_list
    def calc_R(self):
        betas = self.beta_list
        R_list = []
        for j in range(self.n_needlet):
            print(f"calc_R at number:{j}")
            R_nside = self.needlet.at[j, 'nside']
            R = np.zeros((hp.nside2npix(R_nside), self.nmaps, self.nmaps))
            for c1 in range(self.nmaps):
                for c2 in range(c1,self.nmaps):
                    prodMap = betas[j][c1] * betas[j][c2]
                    # hp.mollview(prodMap, norm='hist', title = f"{j = }, {c1 = }, {c2 = }")
                    # plt.show()
                    RMap = hp.smoothing(prodMap, np.deg2rad(self.FWHM[j]),iter=0)
                    # hp.mollview(np.abs(RMap), norm='log',title = f"{c1 = }, {c2 = }")
                    # plt.show()
                    if c1 != c2:
                        R[:,c1,c2] = RMap
                        R[:,c2,c1] = RMap
                    else:
                        # eps = 0.1 * np.min(np.abs(RMap))
                        # R[:,c1,c2] = RMap + eps # for no noise testing
                        # print(f"{eps = }")
                        # R[:,c1,c2] = RMap + np.mean(RMap) # for no noise testing
                        R[:,c1,c2] = RMap
            R_list.append(R)
        self.R = R_list

    def calc_FWHM(self):
        Neff = (self.nmaps - 1) / self.Rtol
        FWHM = np.zeros(self.n_needlet)
        for j in range(self.n_needlet):
            dof = np.sum(self.hl[j]**2 * (2*np.arange(self.lmax+1)+1))
            # dof = ((self.needlet.at[j, 'lmax']+1)**2 - self.needlet.at[j, 'lmin']**2)
            print(f'{dof = }')
            fsky = Neff / dof
            print(f'initial {fsky = }')
            if fsky > 1:
                fsky = 1
            print(f'final {fsky = }')
            dof_eff = fsky * dof
            print(f'{dof_eff = }')
            n_pix = hp.nside2npix(self.needlet.at[j, 'nside'])
            actual_pix = fsky * n_pix
            print(f'the pixel used in {j} scale is:{actual_pix}')
            pixarea = fsky * n_pix * hp.nside2pixarea(self.needlet.at[j, 'nside']) # spherical cap area A=2*pi(1-cos(theta))
            theta = np.arccos(1 - pixarea / (2 * np.pi)) * 180 / np.pi
            FWHM[j] = np.sqrt(8 * np.log(2)) * theta
        self.FWHM = FWHM
    def calc_FWHM_fsky(self, fsky_eff=1/100):
        Neff = (self.nmaps - 1) / self.Rtol
        FWHM = np.zeros(self.n_needlet)
        for j in range(self.n_needlet):
            dof = np.sum(self.hl[j]**2 * (2*np.arange(self.lmax+1)+1))
            # dof = ((self.needlet.at[j, 'lmax']+1)**2 - self.needlet.at[j, 'lmin']**2)
            print(f'{dof = }')
            print(f'initial {fsky_eff = }')
            n_pix = hp.nside2npix(self.needlet.at[j, 'nside'])
            actual_pix = fsky_eff * n_pix
            print(f'the pixel used in {j} scale is:{actual_pix}')
            pixarea = actual_pix * hp.nside2pixarea(self.needlet.at[j, 'nside']) # spherical cap area A=2*pi(1-cos(theta))
            theta = np.arccos(1 - pixarea / (2 * np.pi)) * 180 / np.pi
            FWHM[j] = np.sqrt(8 * np.log(2)) * theta
        self.FWHM = FWHM
    def calc_FWHM_pix_num(self, pix_num=4000):
        Neff = (self.nmaps - 1) / self.Rtol
        FWHM = np.zeros(self.n_needlet)
        for j in range(self.n_needlet):
            dof = np.sum(self.hl[j]**2 * (2*np.arange(self.lmax+1)+1))
            # dof = ((self.needlet.at[j, 'lmax']+1)**2 - self.needlet.at[j, 'lmin']**2)
            print(f'{dof = }')
            n_pix = hp.nside2npix(self.needlet.at[j, 'nside'])
            actual_pix = pix_num
            print(f'the pixel used in {j} scale is:{actual_pix}')
            pixarea = actual_pix * hp.nside2pixarea(self.needlet.at[j, 'nside']) # spherical cap area A=2*pi(1-cos(theta))
            theta = np.arccos(1 - pixarea / (2 * np.pi)) * 180 / np.pi
            FWHM[j] = np.sqrt(8 * np.log(2)) * theta
        self.FWHM = FWHM

    def calc_weight(self, **kwargs):
        oneVec = np.ones(self.nmaps)
        nside = np.array(self.needlet['nside'])
        # self.calc_FWHM_fsky(fsky_eff=0.001)
        # self.calc_FWHM_pix_num(pix_num=6000)
        self.calc_FWHM()
        print(f'{self.FWHM = }')
        self.calc_R()
        R = self.R
        w_list = []
        for j in range(self.n_needlet):
            print(f'calc weight {j}')
            w_R = R[j]
            invR = np.linalg.inv(w_R)
            w = (invR@oneVec).T/(oneVec@invR@oneVec + 1e-15)
            w_list.append(w)
        self.weights = w_list
        print(f'{w = }')

    def calc_ilced_map(self):
        betaNILC = []
        for j in range(self.n_needlet):
            ilc_Beta = self.beta_list[j]
            ilc_w    = self.weights[j]
            res  = np.sum(ilc_Beta * ilc_w, axis=0)
            betaNILC.append(res)

        resMap = 0
        for j in range(self.n_needlet):
            res_alm = hp.map2alm(betaNILC[j])
            print(f'{res_alm.shape = }')
            res_alm = hp.almxfl(res_alm, self.hl[j])
            print(f'resxflalm = {res_alm.shape}')
            ilced_Map = hp.alm2map(res_alm, self.nside)
            resMap = resMap + ilced_Map
        return resMap

    def run_nilc(self):
        print('calc_hl...')
        self.calc_hl()
        print('calc_beta...')
        self.calc_beta()
        if self.weights_config is None:
            print('calc_weight...')
            self.calc_weight()
            np.savez(self.weights_name, *self.weights)
        else:
            print('weight are given...')
            self.weights = []
            for j in range(self.n_needlet):
                weights = np.load(self.weights_config)
                self.weights.append(weights[f'arr_{j}'])
        print('calc_ilced_map...')
        res_map = self.calc_ilced_map()
        return res_map

    def testing(self):
        print(f"{self.nmaps = }")
        print(f"{self.npix = }")
        print(f"{self.nside = }")
        print(f"{self.needlet = }")


sim = np.load(f'../smooth/FULL_SKY/SM_SIM_noPS/{cl_type}/cmbfgnoise.npy')
fg = np.load(f'../smooth/FULL_SKY/SM_FG_noPS/{cl_type}/fg.npy')
# fgnoise = np.load(f'../sim/simdata/{cl_type}/fgnoise.npy')
# cmb = np.load(f'../sim/simdata/{cl_type}/cmb.npy')
noise = np.load(f'../smooth/FULL_SKY/SM_lowNOISE/{cl_type}/noise.npy')

number = 4
needlet_type = 1
obj = NILC(needlet_config=f'./needlets/needlet{needlet_type}.csv', Sm_alms=None, weights_name=f'./FULL_{cl_type}/weight{number}.npz', Sm_maps=sim, lmax=lmax, nside=nside, Rtol=1/100)
# obj = NILC(needlet_config='./needlets/needlet.csv', Sm_alms=None, weights_config=f'./nilcdata/weightexact.npz', Sm_maps=fg, lmax=lmax, nside=nside)

ilc_res = obj.run_nilc()
ilc_cl = hp.anafast(ilc_res,lmax=lmax)
np.save(f'./FULL_{cl_type}/nilc_cl{number}.npy', ilc_cl)

# obj = NILC(needlet_config=f'./needlets/needlet{needlet_type}.csv', Sm_alms=None, weights_config=f'./FULL_{cl_type}/weight{number}.npz', Sm_maps=fg, lmax=lmax, nside=nside)
# fg_res = obj.run_nilc()
# fgres_cl = hp.anafast(fg_res,lmax=lmax)
# np.save(f'./FULL_{cl_type}/nilc_fgres_cl{number}.npy', fgres_cl)

# noise = np.load(f'../smooth/FULL_SKY/SM_NOISE/{cl_type}/noise.npy')
# obj = NILC(needlet_config=f'./needlets/needlet{needlet_type}.csv', Sm_alms=None, weights_config=f'./FULL_{cl_type}/weight{number}.npz', Sm_maps=noise, lmax=lmax, nside=nside)
# noise_res = obj.run_nilc()
# noiseres_cl = hp.anafast(noise_res,lmax=lmax)
# np.save(f'./FULL_B/NOISE/nilc_noise_cl{number}{i}.npy', noiseres_cl)


# for i in range(15,30):
#     print(f'{i}')
#     noise = np.load(f'../smooth/FULL_SKY/SM_NOISE/{i}/{cl_type}/noise.npy')
#     obj = NILC(needlet_config=f'./needlets/needlet{needlet_type}.csv', Sm_alms=None, weights_config=f'./FULL_{cl_type}/weight{number}.npz', Sm_maps=noise, lmax=lmax, nside=nside)
#     noise_res = obj.run_nilc()
#     noiseres_cl = hp.anafast(noise_res,lmax=lmax)
#     np.save(f'./FULL_B/NOISE/nilc_noise_cl{number}{i}.npy', noiseres_cl)



# obj = NILC(needlet_config=f'./needlets/needlet{needlet_type}.csv', Sm_alms=None, weights_config=f'./{cl_type}/weight{number}.npz', Sm_maps=fgnoise, lmax=lmax, nside=nside)
# d = obj.run_nilc()
# obj = NILC(needlet_config=f'./needlets/needlet{needlet_type}.csv', Sm_alms=None, weights_config=f'./{cl_type}/weight{number}.npz', Sm_maps=cmb, lmax=lmax, nside=nside)
# s = obj.run_nilc()

# cl_sd = 2 * hp.anafast(s,d,lmax=lmax)
# np.save(f'./{cl_type}/nilc_clsd{number}.npy', cl_sd)

