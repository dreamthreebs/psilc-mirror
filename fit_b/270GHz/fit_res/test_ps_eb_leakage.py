import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import pymaster as nmt
import os,sys

from pathlib import Path
config_dir = Path(__file__).parent.parent
print(f'{config_dir=}')
sys.path.insert(0, str(config_dir))
from config import freq, lmax, nside, beam
l = np.arange(lmax+1)

rlz_idx=0
threshold = 3

df = pd.read_csv('../../../FGSim/FreqBand')
print(f'{freq=}, {beam=}')

def calc_fsky(mask):
    """
    Return the effective sky fraction f_sky = mean(mask**2).

    Works for binary or apodized HEALPix masks.
    """
    return np.sum(mask**2) / np.size(mask)


# bin_mask = np.load('../../../src/mask/north/BINMASKG2048.npy')
bin_mask = np.load('../../../psfit/fitv4/fit_res/2048/ps_mask/new_mask/BIN_C1_3_C1_3.npy')
apo_mask = np.load('../../../psfit/fitv4/fit_res/2048/ps_mask/new_mask/apo_C1_3_apo_3_apo_3.npy')
print(f'{np.sum(apo_mask)/np.size(apo_mask)=}')
# ps_mask = np.load(f'../inpainting/new_mask/apo_ps_mask.npy')
# ps_mask_2deg = np.load(f'../inpainting/new_mask/apo_ps_mask_2degree.npy')
ps_mask_25 = np.load(f"../inpainting/new_mask/apo_ps_mask_25.npy")
# hp.orthview(apo_mask, rot=[100,50,0], half_sky=True)
# hp.orthview(ps_mask, rot=[100,50,0], half_sky=True)
# hp.orthview(ps_mask_2deg, rot=[100,50,0], half_sky=True)
# hp.orthview(ps_mask_25, rot=[100,50,0], half_sky=True)
# plt.show()

fsky_apo = calc_fsky(mask=apo_mask)
# fsky_ps_mask = calc_fsky(mask=ps_mask)
# fsky_ps_2deg = calc_fsky(mask=ps_mask_2deg)
fsky_ps_25 = calc_fsky(mask=ps_mask_25)
print(f"{fsky_apo=}")
# print(f"{fsky_ps_mask=}")
# print(f"{fsky_ps_2deg=}")
print(f"{fsky_ps_25=}")

noise_seed = np.load('../../seeds_noise_2k.npy')
cmb_seed = np.load('../../seeds_cmb_2k.npy')
fg_seed = np.load('../../seeds_fg_2k.npy')


# utils
def calc_lmax(beam):
    lmax_eff = 2 * np.pi / np.deg2rad(beam) * 60
    print(f'{lmax_eff=}')
    return int(lmax_eff) + 1

def find_left_nearest_index_np(arr, target):
    # Find the indices of values less than or equal to the target
    valid_indices = np.where(arr <= target)[0]

    # If there are no valid indices, handle the case (e.g., return None)
    if valid_indices.size == 0:
        return None

    # Get the index of the largest value less than or equal to the target
    nearest_index = valid_indices[-1]  # The largest valid index
    return nearest_index + 1


def generate_bins(l_min_start=30, delta_l_min=30, l_max=1500, fold=0.3, l_threshold=None):
    bins_edges = []
    l_min = l_min_start  # starting l_min

    # Fixed binning until l_threshold if provided
    if l_threshold is not None:
        while l_min < l_threshold:
            l_next = l_min + delta_l_min
            if l_next > l_threshold:
                break
            bins_edges.append(l_min)
            l_min = l_next

    # Transition to dynamic binning
    while l_min < l_max:
        delta_l = max(delta_l_min, int(fold * l_min))
        l_next = l_min + delta_l
        bins_edges.append(l_min)
        l_min = l_next

    # Adding l_max to ensure the last bin goes up to l_max
    bins_edges.append(l_max)
    return bins_edges[:-1], bins_edges[1:]

def calc_dl_from_pol_map(m_q, m_u, bl, apo_mask, bin_dl, masked_on_input, purify_b):
    f2p = nmt.NmtField(apo_mask, [m_q, m_u], beam=bl, masked_on_input=masked_on_input, purify_b=purify_b, lmax=lmax, lmax_mask=lmax)
    w22p = nmt.NmtWorkspace.from_fields(f2p, f2p, bin_dl)
    # dl = nmt.workspaces.compute_full_master(pol_field, pol_field, b=bin_dl)
    dl = w22p.decouple_cell(nmt.compute_coupled_cell(f2p, f2p))[3]
    return dl

def calc_dl_from_scalar_map(scalar_map, bl, apo_mask, bin_dl, masked_on_input):
    scalar_field = nmt.NmtField(apo_mask, [scalar_map], beam=bl, masked_on_input=masked_on_input, lmax=lmax, lmax_mask=lmax)
    dl = nmt.compute_full_master(scalar_field, scalar_field, bin_dl)
    return dl[0]

def gen_fg_cl():
    cl_fg = np.load('../data/debeam_full_b/cl_fg.npy')
    Cl_TT = cl_fg[0]
    Cl_EE = cl_fg[1]
    Cl_BB = cl_fg[2]
    Cl_TE = np.zeros_like(Cl_TT)
    return np.array([Cl_TT, Cl_EE, Cl_BB, Cl_TE])

def gen_map(rlz_idx=0, mode='mean', return_noise=False):
    # mode can be mean or std
    nside = 2048

    nstd = np.load(f'../../../FGSim/NSTDNORTH/2048/{freq}.npy')
    npix = hp.nside2npix(nside=2048)
    np.random.seed(seed=noise_seed[rlz_idx])
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    print(f"{np.std(noise[1])=}")

    if return_noise:
        return noise

    ps = np.load(f'../../../fitdata/2048/PS/{freq}/ps.npy')
    fg = np.load(f'../../../fitdata/2048/FG/{freq}/fg.npy')

    cls = np.load('../../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    if mode=='std':
        np.random.seed(seed=cmb_seed[rlz_idx])
    elif mode=='mean':
        np.random.seed(seed=cmb_seed[0])

    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=3*nside-1)

    pcfn = noise + ps + cmb_iqu + fg
    cfn = noise + cmb_iqu + fg
    cf = cmb_iqu + fg
    n = noise
    return pcfn, cfn, cf, n

def gen_cmb(rlz_idx=0, mode="std"):
    cls = np.load('../../../src/cmbsim/cmbdata/cmbcl_8k.npy')

    if mode=='std':
        np.random.seed(seed=cmb_seed[rlz_idx])
    elif mode=='mean':
        np.random.seed(seed=cmb_seed[0])

    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=3*nside-1)
    return cmb_iqu


# initialize the band power
bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]
# l_min_edges, l_max_edges = generate_bins(l_min_start=10, delta_l_min=30, l_max=lmax+1, fold=0.2)
l_min_edges, l_max_edges = generate_bins(l_min_start=42, delta_l_min=40, l_max=lmax+1, fold=0.1, l_threshold=400)
# delta_ell = 30
# bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
# bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=40, is_Dell=True)
bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
ell_arr = bin_dl.get_effective_ells()
print(f'{ell_arr=}')


def check_ps_mask_res():
    cmb = gen_cmb(rlz_idx)

    # print("begin no_ps_mask")
    # dl_no_ps_mask = calc_dl_from_pol_map(m_q=cmb[1], m_u=cmb[2], bl=bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    # print("begin 1deg_ps_mask")
    # dl_1deg_ps_mask = calc_dl_from_pol_map(m_q=cmb[1], m_u=cmb[2], bl=bl, apo_mask=ps_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    # print("begin 2deg_ps_mask")
    # dl_2deg_ps_mask = calc_dl_from_pol_map(m_q=cmb[1], m_u=cmb[2], bl=bl, apo_mask=ps_mask_2deg, bin_dl=bin_dl, masked_on_input=False, purify_b=True)

    print("begin only 25 point sources")
    dl_25_ps_mask = calc_dl_from_pol_map(m_q=cmb[1], m_u=cmb[2], bl=bl, apo_mask=ps_mask_25, bin_dl=bin_dl, masked_on_input=False, purify_b=True)

    path_dl_qu_mask = Path(f'BIAS/test_ps_eb_leakage')
    path_dl_qu_mask.mkdir(parents=True, exist_ok=True)

    np.save(path_dl_qu_mask / Path(f'25_ps_mask_{rlz_idx}.npy'), dl_25_ps_mask)
    # np.save(path_dl_qu_mask / Path(f'no_ps_mask_{rlz_idx}.npy'), dl_no_ps_mask)
    # np.save(path_dl_qu_mask / Path(f'1deg_ps_mask_{rlz_idx}.npy'), dl_1deg_ps_mask)
    # np.save(path_dl_qu_mask / Path(f'2deg_ps_mask_{rlz_idx}.npy'), dl_2deg_ps_mask)


def plot_ps_mask_res():
    # initialize the band power
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]
    # l_min_edges, l_max_edges = generate_bins(l_min_start=10, delta_l_min=30, l_max=lmax+1, fold=0.2)
    l_min_edges, l_max_edges = generate_bins(l_min_start=42, delta_l_min=40, l_max=lmax+1, fold=0.1, l_threshold=400)
    # delta_ell = 30
    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=40, is_Dell=True)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()
    print(f'{ell_arr=}')


    lmax_eff = calc_lmax(beam=beam)
    lmax_ell_arr = find_left_nearest_index_np(ell_arr, target=lmax_eff)
    print(f'{ell_arr=}')
    ell_arr = ell_arr[:lmax_ell_arr]
    print(f'{ell_arr[:lmax_ell_arr]=}')
    print(f'{lmax_ell_arr=}')
    cl_cmb = np.load('/afs/ihep.ac.cn/users/w/wangyiming25/work/dc2/psilc/src/cmbsim/cmbdata/cmbcl_8k.npy').T
    print(f'{cl_cmb.shape=}')
    l = np.arange(lmax_eff+1)
    dl_in = bin_dl.bin_cell(cl_cmb[2,:lmax+1])

    rlz_range = np.arange(1, 200)

    no_ps_mask_list = [
        np.load(f'./BIAS/test_ps_eb_leakage/no_ps_mask_{rlz_idx}.npy')
        for rlz_idx in rlz_range
    ]

    deg_1_ps_mask_list = [
        np.load(f'./BIAS/test_ps_eb_leakage/1deg_ps_mask_{rlz_idx}.npy')
        for rlz_idx in rlz_range
    ]

    deg_2_ps_mask_list = [
        np.load(f'./BIAS/test_ps_eb_leakage/2deg_ps_mask_{rlz_idx}.npy')
        for rlz_idx in rlz_range
    ]

    ps_25_ps_mask_list = [
        np.load(f'./BIAS/test_ps_eb_leakage/25_ps_mask_{rlz_idx}.npy')
        for rlz_idx in rlz_range
    ]



    no_ps_mask_mean = np.mean(no_ps_mask_list, axis=0)
    deg_1_ps_mask_mean = np.mean(deg_1_ps_mask_list, axis=0)
    deg_2_ps_mask_mean = np.mean(deg_2_ps_mask_list, axis=0)
    ps_25_ps_mask_mean = np.mean(ps_25_ps_mask_list, axis=0)

    no_ps_mask_std = np.std(no_ps_mask_list, axis=0)
    deg_1_ps_mask_std = np.std(deg_1_ps_mask_list, axis=0)
    deg_2_ps_mask_std = np.std(deg_2_ps_mask_list, axis=0)
    ps_25_ps_mask_std = np.std(ps_25_ps_mask_list, axis=0)


    plt.figure(1)
    plt.errorbar(ell_arr*0.990, ps_25_ps_mask_mean[:lmax_ell_arr], ps_25_ps_mask_std[:lmax_ell_arr], label='only 25 point sources 1deg', marker='.')
    plt.errorbar(ell_arr*0.995, no_ps_mask_mean[:lmax_ell_arr], no_ps_mask_std[:lmax_ell_arr], label='only edge mask', marker='.')
    plt.plot(ell_arr*1, dl_in[:lmax_ell_arr], label='CMB input', marker='.', color='black')
    plt.errorbar(ell_arr*1.005, deg_1_ps_mask_mean[:lmax_ell_arr], deg_1_ps_mask_std[:lmax_ell_arr], label='add ps 1deg mask', marker='.')
    plt.errorbar(ell_arr*1.01, deg_2_ps_mask_mean[:lmax_ell_arr], deg_2_ps_mask_std[:lmax_ell_arr], label='add ps 2deg mask', marker='.')
    plt.loglog()
    plt.legend()
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$D_\ell^{BB}$")
    plt.title('EB leakage correction')
    plt.show()


    plt.figure(2)
    plt.plot(ell_arr, ps_25_ps_mask_std[:lmax_ell_arr], label='ps 25 mask', marker='.', color='black')
    plt.plot(ell_arr, no_ps_mask_std[:lmax_ell_arr], label='only edge mask', marker='.', color='black')
    plt.plot(ell_arr, deg_1_ps_mask_std[:lmax_ell_arr], label='add ps 1deg mask', marker='.')
    plt.plot(ell_arr, deg_2_ps_mask_std[:lmax_ell_arr], label='add ps 2deg mask', marker='.')
    plt.legend()
    plt.loglog()

    plt.title('standard deviaion comparison')
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$D_\ell^{BB}$")

    plt.show()


# check_ps_mask_res()
plot_ps_mask_res()
