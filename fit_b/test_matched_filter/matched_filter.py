import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from pathlib import Path

class MatchedFilter:
    ''' usr guide:
    1. Calculate wl. Then check it or not
    '''
    def __init__(self, nside, lmax, beam, cl_tot, pol=False, name='default'):
        self.nside = nside
        self.lmax = lmax
        self.beam = beam # in arcmin
        self.name = name


        if (len(cl_tot) < (lmax+1)):
            raise ValueError('the length cl is smaller than lmax, try other input lmax!')

        if pol: # if True, use polarization beam
            self.bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:lmax+1, 1]
        else:
            self.bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)

        self.cl_tot = cl_tot[:lmax+1] # The power spectrum at some frequency except point sources, including diffuse foreground, instrumental noise, CMB ...

    def calc_wl(self, normalize):
        # normalize only need your input cl total, it will generate a point source map and then get the normalize factor to normalize wl and make wl to be an unbiased filter.
        print(f'calculating wl...')

        self.wl = self.bl / self.cl_tot
        self.wl[0:2] = 0

        if normalize:
            print(f'normalizing...')
            npix = hp.nside2npix(self.nside)

            ipix_ctr = hp.ang2pix(theta=0, phi=0, lonlat=True, nside=self.nside)
            print(f'{ipix_ctr=}')

            delta_m = np.zeros(npix)
            flux_I = 1000
            delta_m[ipix_ctr] = flux_I

            sm_m = hp.smoothing(delta_m,beam_window=self.bl, pol=False)
            ps_max = np.max(sm_m)

            ps_out = hp.smoothing(sm_m, beam_window=self.wl, pol=False)
            ps_out_max = np.max(ps_out)

            self.wl = self.wl * ps_max / ps_out_max

            print(f'normalized')
            path_wl = Path(f'./mf_data')
            path_wl.mkdir(exist_ok=True, parents=True)
            np.save(path_wl / Path(f'normalized_wl_{self.name}.npy'), self.wl)

            # ps_after_norm = hp.smoothing(sm_m, beam_window=self.wl, pol=False)
            # hp.gnomview(sm_m, title='ps input')
            # hp.gnomview(ps_out, title='ps output')
            # hp.gnomview(ps_after_norm, title='ps_after_norm')
            # plt.show()


        return self.wl

    def check_all_cl(self):
        # l = np.arange(lmax + 1)

        plt.loglog(self.bl, label='bl')
        plt.loglog(self.cl_tot, label='cl tot')
        plt.loglog(self.wl, label='wl')
        plt.legend()
        plt.show()

    def run_mf(self, m_obs, m_tot):
        file_wl = Path(f'./mf_data/normalized_wl_{self.name}.npy')
        if file_wl.exists():
            self.wl = np.load(file_wl)
        if not hasattr(self, "wl"):
            self.calc_wl(normalize=True)

        obs_out = hp.smoothing(m_obs, beam_window=self.wl, pol=False)
        tot_out = hp.smoothing(m_tot, beam_window=self.wl, pol=False)

        sigma = np.std(tot_out)
        snr =  obs_out / sigma

        print(f'{sigma=}')

        # ps = np.load('./data/ps_map.npy')
        # hp.gnomview(ps, title='ps')
        # hp.gnomview(m_obs, title='input observation')
        # hp.gnomview(m_tot, title='input total')
        # hp.gnomview(obs_out, title='filtered observation')
        # hp.gnomview(tot_out, title='filtered total')
        # hp.gnomview(snr, title='snr')
        # snr[snr<5] = 0
        # hp.gnomview(snr, title='snr > 5')
        # plt.show()
        return obs_out, tot_out, snr, sigma, self.wl



def test_mf():
    nside = 1024
    lmax = 2500
    beam = 11

    pcn = np.load('./data/pcn.npy')
    cn = np.load('./data/cn.npy')
    cl_cn = hp.anafast(cn, lmax=lmax)

    obj_mf = MatchedFilter(nside, lmax, beam, cl_tot=cl_cn)

    obj_mf.calc_wl(normalize=True)
    obj_mf.check_all_cl()

    # obj_mf.run_mf(m_obs=pcn, m_tot=cn)

if __name__ == "__main__":

    test_mf()

