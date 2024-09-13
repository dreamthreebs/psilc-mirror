import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from pathlib import Path
from eblc_base_slope import EBLeakageCorrection

nside = 512
npix = hp.nside2npix(nside=nside)
beam = 67
mask = np.load('../../src/mask/north/BINMASKG.npy')
# m = np.load('../../fitdata/synthesis_data/2048/CMBNOISE/270/1.npy')
rlz_idx = 0
cmb_seed = np.load('../seeds_cmb_2k.npy')
noise_seed = np.load('../seeds_noise_2k.npy')
fg_seed = np.load('../seeds_fg_2k.npy')

def gen_fg_cl():
    Cl_TT = np.load('../Cl_fg/data/cl_fg_TT.npy')
    Cl_EE = np.load('../Cl_fg/data/cl_fg_EE.npy')
    Cl_BB = np.load('../Cl_fg/data/cl_fg_BB.npy')
    Cl_TE = np.zeros_like(Cl_TT)
    return np.array([Cl_TT, Cl_EE, Cl_BB, Cl_TE])

def gen_map(lmax, component):

    if component == 'c':
        cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy').T
        np.random.seed(seed=cmb_seed[rlz_idx])
        cmb_iqu = hp.synfast(cls=cls, nside=nside, fwhm=np.deg2rad(beam)/60, lmax=3*nside-1, new=True)
        return cmb_iqu

    elif component == 'cn':
        cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy').T
        np.random.seed(seed=cmb_seed[rlz_idx])
        cmb_iqu = hp.synfast(cls=cls, nside=nside, fwhm=np.deg2rad(beam)/60, lmax=3*nside-1, new=True)

        nstd = np.load('../../FGSim/NSTDNORTH/512/30.npy')
        np.random.seed(seed=noise_seed[rlz_idx])
        noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
        return cmb_iqu + noise

    elif component == 'n':
        nstd = np.load('../../FGSim/NSTDNORTH/512/30.npy')
        np.random.seed(seed=noise_seed[rlz_idx])
        noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
        return noise

    elif component == 'f':
        cls_fg = gen_fg_cl()
        np.random.seed(seed=fg_seed[rlz_idx])
        m_fg = hp.synfast(cls=cls_fg, nside=nside, fwhm=0, new=True, lmax=600)
        return m_fg

    elif component == 'cfn':
        cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy').T
        np.random.seed(seed=cmb_seed[rlz_idx])
        cmb_iqu = hp.synfast(cls=cls, nside=nside, fwhm=np.deg2rad(beam)/60, lmax=3*nside-1, new=True)

        nstd = np.load('../../FGSim/NSTDNORTH/512/30.npy')
        np.random.seed(seed=noise_seed[rlz_idx])
        noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))

        cls_fg = gen_fg_cl()
        np.random.seed(seed=fg_seed[rlz_idx])
        m_fg = hp.synfast(cls=cls_fg, nside=nside, fwhm=0, new=True, lmax=600)
        return m_fg + noise + cmb_iqu


m_c = gen_map(lmax=1000, component='c')
m_n = gen_map(lmax=1000, component='n')
m_f = gen_map(lmax=1000, component='f')
m_cfn = gen_map(lmax=1000, component='cfn')


print(f'method=cutqufitqu')
print(f'eblc lmax=600')

lmax = 600
obj_c = EBLeakageCorrection(m=m_c, lmax=lmax, nside=nside, mask=mask, post_mask=mask)
_,_, cln_c = obj_c.run_eblc()

obj_n = EBLeakageCorrection(m=m_n, lmax=lmax, nside=nside, mask=mask, post_mask=mask)
_,_, cln_n = obj_n.run_eblc()

obj_f = EBLeakageCorrection(m=m_f, lmax=lmax, nside=nside, mask=mask, post_mask=mask)
_,_, cln_f = obj_f.run_eblc()

obj_cfn = EBLeakageCorrection(m=m_cfn, lmax=lmax, nside=nside, mask=mask, post_mask=mask)
_,_, cln_cfn = obj_cfn.run_eblc()


hp.orthview(cln_c, rot=[100,50,0], title='c')
hp.orthview(cln_n, rot=[100,50,0], title='n')
hp.orthview(cln_f, rot=[100,50,0], title='f')
hp.orthview(cln_cfn, rot=[100,50,0], title='cfn')
hp.orthview(cln_cfn - cln_c - cln_n - cln_f, rot=[100,50,0], title='linear?')
plt.show()

path_data = Path('./data/check_bias')
path_data.mkdir(exist_ok=True, parents=True)
np.save(path_data / Path(f'cfn.npy'), cln_cfn)
np.save(path_data / Path(f'c.npy'), cln_c)
np.save(path_data / Path(f'f.npy'), cln_f)
np.save(path_data / Path(f'n.npy'), cln_n)

