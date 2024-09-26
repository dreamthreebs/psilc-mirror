import numpy as np
import healpy as hp
import pymaster as nmt
import matplotlib.pyplot as plt

from pathlib import Path

lmax = 1000
nside = 2048
beam = 67
rlz_idx=10
fg = np.load('../../fitdata/2048/FG/30/fg.npy')
apo_mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')
bin_mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
cmb_seed = np.load('../seeds_cmb_2k.npy')
noise_seed = np.load('../seeds_noise_2k.npy')
fg_seed = np.load('../seeds_fg_2k.npy')

def gen_map(lmax, component):

    if component == 'c':
        cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy').T
        np.random.seed(seed=cmb_seed[rlz_idx])
        cmb_iqu = hp.synfast(cls=cls, nside=nside, fwhm=np.deg2rad(beam)/60, lmax=3*nside-1, new=True)
        return cmb_iqu

    elif component == '10c':
        cls = 10 * np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy').T
        np.random.seed(seed=cmb_seed[rlz_idx])
        cmb_iqu = hp.synfast(cls=cls, nside=nside, fwhm=np.deg2rad(beam)/60, lmax=3*nside-1, new=True)
        return cmb_iqu

    elif component == '1000c':
        cls = 1000 * np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy').T
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


def generate_bins(l_min_start=30, delta_l_min=30, l_max=1500, fold=0.3):
    bins_edges = []
    l_min = l_min_start  # starting l_min

    while l_min < l_max:
        delta_l = max(delta_l_min, int(fold * l_min))
        l_next = l_min + delta_l
        bins_edges.append(l_min)
        l_min = l_next

    # Adding l_max to ensure the last bin goes up to l_max
    bins_edges.append(l_max)
    return bins_edges[:-1], bins_edges[1:]

def calc_dl_from_pol_map(m_q, m_u, bl, apo_mask, bin_dl, masked_on_input, purify_b):
    pol_field = nmt.NmtField(apo_mask, [m_q, m_u], beam=bl, masked_on_input=masked_on_input, purify_b=purify_b)
    dl = nmt.compute_full_master(pol_field, pol_field, bin_dl)
    return dl[3]

def calc_dl_from_scalar_map(scalar_map, bl, apo_mask, bin_dl, masked_on_input):
    scalar_field = nmt.NmtField(apo_mask, [scalar_map], beam=bl, masked_on_input=masked_on_input)
    dl = nmt.compute_full_master(scalar_field, scalar_field, bin_dl)
    return dl[0]

def get_dl_pol():
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=20, l_max=lmax, fold=0.2)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()
    cmb = gen_map(lmax=1000, component='c')

    dl_cmb = calc_dl_from_pol_map(m_q=cmb[1], m_u=cmb[2], bl=bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    
    dl_data_path = Path('nmt_cmb_data/pol')
    dl_data_path.mkdir(exist_ok=True, parents=True)
    np.save(dl_data_path / Path(f'dl_cmb_{rlz_idx}.npy'), dl_cmb)

# get_dl_pol()

def get_dl_sca():
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=20, l_max=lmax, fold=0.2)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    cmb = gen_map(lmax=1000, component='c')
    m_cmb = hp.alm2map(hp.map2alm(cmb)[2], nside=nside)

    dl_cmb = calc_dl_from_scalar_map(scalar_map=m_cmb, bl=bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)
    
    dl_data_path = Path('nmt_cmb_data/sca')
    dl_data_path.mkdir(exist_ok=True, parents=True)
    np.save(dl_data_path / Path(f'dl_cmb_{rlz_idx}.npy'), dl_cmb)

# get_dl_sca()

def check_cmb_dl():
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=20, l_max=lmax, fold=0.2)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    l = np.arange(lmax+1)
    cl_hp = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy').T

    dl_pol = np.load(f'./nmt_cmb_data/pol/dl_cmb_{rlz_idx}.npy')
    dl_sca = np.load(f'./nmt_cmb_data/sca/dl_cmb_{rlz_idx}.npy')

    plt.loglog(ell_arr, dl_pol, label='pol')
    plt.loglog(ell_arr, dl_sca, label='sca')
    plt.loglog(l, l*(l+1)*cl_hp[2,:lmax+1]/(2*np.pi), label='input')
    plt.legend()
    plt.show()


# check_cmb_dl()

def check_cmb_dl_avg():
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=20, l_max=lmax, fold=0.2)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    l = np.arange(lmax+1)
    cl_hp = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy').T

    dl_pol_list = []
    dl_sca_list = []
    for rlz_idx in range(200):
        dl_pol = np.load(f'./nmt_cmb_data/pol/dl_cmb_{rlz_idx}.npy')
        dl_sca = np.load(f'./nmt_cmb_data/sca/dl_cmb_{rlz_idx}.npy')
        dl_pol_list.append(dl_pol)
        dl_sca_list.append(dl_sca)

    pol_mean = np.mean(dl_pol_list, axis=0)
    sca_mean = np.mean(dl_sca_list, axis=0)

    plt.loglog(ell_arr, pol_mean, label='pol')
    plt.loglog(ell_arr, sca_mean, label='sca')
    plt.loglog(l, l*(l+1)*cl_hp[2,:lmax+1]/(2*np.pi), label='input')
    plt.legend()
    plt.show()

# check_cmb_dl_avg()

