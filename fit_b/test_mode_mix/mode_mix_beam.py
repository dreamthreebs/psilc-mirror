import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt

from pathlib import Path

nside = 1024
lmax = 3 * nside - 1
l = np.arange(lmax + 1)

rlz_idx=0

fg_seeds = np.load('../seeds_fg_2k.npy')
beam = 67

def gen_fg_cl():
    l = np.arange(8001)

    cl_TT = 10 / (l)**2.2
    cl_TT[0:2] = 0
    cl_TE = 2 / (l)**2.2
    cl_TE[0:2] = 0
    cl_EE = 100 / (l)**2.2
    cl_EE[0:2] = 0
    cl_BB = 10 / (l)**2.2
    cl_BB[0:2] = 0

    # m_1 = hp.synfast([cl_TT, cl_EE, cl_BB, cl_TE], nside=nside, new=True)
    # cl_1 = hp.anafast(m_1, lmax=lmax)

    # cl_TT = 1 / (l+10)
    # cl_TE = 1 / (l+20)
    # cl_EE = 0.8 / (l+30)
    # cl_BB = 0.6 / (l+40)

    # m_2 = hp.synfast([cl_TT, cl_EE, cl_BB, cl_TE], nside=nside, new=True)
    # cl_2 = hp.anafast(m_2, lmax=lmax)

    # path_data = Path('./data/cl_fg')
    # path_data.mkdir(exist_ok=True, parents=True)
    # np.save(path_data / 'cl_1.npy', cl_1)
    # np.save(path_data / 'cl_2.npy', cl_2)

    plt.plot(l, l*(l+1)*cl_TT, label='TT')
    plt.plot(l, l*(l+1)*cl_TE, label='TE')
    plt.plot(l, l*(l+1)*cl_EE, label='EE')
    plt.plot(l, l*(l+1)*cl_BB, label='BB')
    plt.loglog()
    plt.legend()
    plt.show()

    np.save('./data/cl_fg/cl_fg_10010.npy', np.asarray([cl_TT, cl_EE, cl_BB, cl_TE]))

# gen_fg_cl()


def test_fg():
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]
    cls_fg = np.load('./data/cl_fg/cl_fg_86.npy')
    np.random.seed(seed=fg_seeds[rlz_idx])
    fg_rlz = hp.synfast(cls_fg, nside=nside, new=True, fwhm=np.deg2rad(beam)/60)

    cls_rlz = hp.anafast(fg_rlz, lmax=lmax)

    # path_cls = Path('./data/cl_rlz')
    # path_cls.mkdir(exist_ok=True, parents=True)
    # np.save(path_cls / Path(f'{rlz_idx}.npy'), cls_rlz)

    plt.plot(l*(l+1)*cls_fg[2,:lmax+1]/(2*np.pi), label='input B')
    plt.plot(l*(l+1)*cls_fg[1,:lmax+1]/(2*np.pi), label='input E')
    # plt.plot(l*(l+1)*cls_rlz[2]/bl**2/(2*np.pi), label='one realization')
    plt.semilogy()
    plt.ylim(1e-5,1e5)
    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell$')
    plt.legend()
    plt.show()

# test_fg()

def check_cmb():

    cls_cmb = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy').T
    cls_list = []
    for i in range(1000):
        cls_rlz = np.load(f'./data/cl_rlz/{i}.npy')[2]
        # print(f'{cls_rlz.shape=}')
        cls_list.append(cls_rlz)
    cls_mean = np.mean(cls_list, axis=0)

    plt.figure(1)
    plt.plot(l*(l+1)*cls_cmb[2,:lmax+1]/(2*np.pi), label='input')
    plt.plot(l*(l+1)*cls_mean/(2*np.pi), label='realization mean')
    plt.legend()
    plt.title('power spectrum')
    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell^{BB}$')
    plt.show()

    plt.figure(2)
    plt.plot((cls_cmb[2,:lmax+1] - cls_mean)/cls_cmb[2,:lmax+1])
    plt.title('relative error')
    plt.xlabel('$\\ell$')
    plt.ylabel('(exp-input) / input')
    plt.show()

# check_cmb()

########### check_B mode mixing

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

def get_fg():
    cls_fg = np.load('./data/cl_fg/cl_fg_1010.npy')
    np.random.seed(seed=fg_seeds[rlz_idx])
    fg_rlz = hp.synfast(cls_fg, nside=nside, new=True, fwhm=np.deg2rad(beam)/60)
    return fg_rlz


def calc_dl_from_scalar_map(scalar_map, apo_mask, bin_dl, masked_on_input):
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,2]
    scalar_field = nmt.NmtField(apo_mask, [scalar_map], beam=bl, masked_on_input=masked_on_input)
    dl = nmt.compute_full_master(scalar_field, scalar_field, bin_dl)
    return dl[0]

def calc_dl_from_pol_map(m_q, m_u, apo_mask, bin_dl, masked_on_input, purify_b):
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,2]
    pol_field = nmt.NmtField(apo_mask, [m_q, m_u], beam=bl, masked_on_input=masked_on_input, purify_b=purify_b)
    dl = nmt.compute_full_master(pol_field, pol_field, bin_dl)
    return dl[3]

def test_mode_mix():
    # l_min_edges, l_max_edges = generate_bins(l_min_start=4, delta_l_min=20, l_max=lmax, fold=0.3)
    # bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True) # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=20, is_Dell=True)
    bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=20, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    bin_mask = np.load('../../src/mask/north/BINMASKG1024.npy')
    apo_mask = nmt.mask_apodization(mask_in=bin_mask, aposize=5)

    m = get_fg()
    dl = calc_dl_from_pol_map(m_q=m[1]*bin_mask, m_u=m[2]*bin_mask, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]
    cl_no_mask = hp.anafast(m, lmax=lmax)[2] / bl**2

    cls_fg = np.load('./data/cl_fg/cl_fg_1010.npy')
    dl_true = bin_dl.bin_cell(cls_fg[2,:lmax+1])
    dl_no_mask = bin_dl.bin_cell(cl_no_mask)
    print(f'{dl_true=}')

    path_dl_rlz = Path('./data/dls_beam_1010')
    path_dl_rlz.mkdir(exist_ok=True, parents=True)
    np.save(path_dl_rlz / Path(f'full_{rlz_idx}.npy'), dl_no_mask)
    np.save(path_dl_rlz / Path(f'partial_{rlz_idx}.npy'), dl)

    # plt.figure(1)
    # # plt.plot(l*(l+1)*cls_cmb[0,:lmax+1]/(2*np.pi), label='input')
    # plt.plot(ell_arr, dl_true, label='input')
    # plt.plot(ell_arr, dl_no_mask, label='full sky SHT')
    # plt.plot(ell_arr, dl, label='realization')
    # plt.legend()
    # plt.title('power spectrum')
    # plt.xlabel('$\\ell$')
    # plt.ylabel('$D_\\ell^{TT}$')
    # plt.show()

# test_mode_mix()

def check_mode_mix():

    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=20, is_Dell=True)
    bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=20, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    cls_fg = np.load('./data/cl_fg/cl_fg_68.npy')
    dl_true = bin_dl.bin_cell(cls_fg[2,:lmax+1])

    dl_full_list = []
    dl_partial_list = []
    for i in range(500):
        dl_full = np.load(f'./data/dls_beam_68/full_{i}.npy')
        dl_partial = np.load(f'./data/dls_beam_68/partial_{i}.npy')
        print(f'{dl_full=}')
        print(f'{dl_partial=}')

        dl_full_list.append(dl_full)
        dl_partial_list.append(dl_partial)

    dl_full_arr = np.array(dl_full_list)
    dl_partial_arr = np.array(dl_partial_list)

    dl_full_mean = np.mean(dl_full_arr, axis=0)
    dl_partial_mean = np.mean(dl_partial_arr, axis=0)

    plt.figure(1)
    # plt.plot(l*(l+1)*cls_cmb[0,:lmax+1]/(2*np.pi), label='input')
    plt.plot(ell_arr, dl_true, label='input')
    plt.plot(ell_arr, dl_partial_mean, label='partial sky Namaster')
    plt.plot(ell_arr, dl_full_mean, label='full sky Namaster')
    plt.semilogy()

    plt.legend()
    plt.title('power spectrum')
    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell^{BB}$')
    plt.ylim(1e-5,1e5)

    plt.figure(2)
    plt.plot(ell_arr, np.abs(dl_partial_mean - dl_true)/dl_true, label='partial relative error')
    plt.plot(ell_arr, np.abs(dl_full_mean - dl_true)/dl_true, label='full relative error')
    plt.legend()
    # plt.semilogy()
    plt.ylim(0,1)
    plt.title('power spectrum')
    plt.xlabel('$\\ell$')
    plt.ylabel('abs(exp-true)/true')


    plt.show()

    # path_save = Path('/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20240926')
    # path_save.mkdir(exist_ok=True, parents=True)
    # plt.savefig(path_save / Path('mean.png'), dpi=300)


# check_mode_mix()

def cpr_mode_mix():

    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=20, is_Dell=True)
    bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=20, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    cls_fg = np.load('./data/cl_fg/cl_fg_1010.npy')
    dl_true = bin_dl.bin_cell(cls_fg[2,:lmax+1])

    dl_full_list = []
    dl_partial_list = []
    dl_no_beam_full_list = []
    dl_no_beam_partial_list = []
    for i in range(200):
        dl_full = np.load(f'./data/dls_beam_1010/full_{i}.npy')
        dl_partial = np.load(f'./data/dls_beam_1010/partial_{i}.npy')
        dl_no_beam_full = np.load(f'./data/dls_fg_1010/full_{i}.npy')
        dl_no_beam_partial = np.load(f'./data/dls_fg_1010/partial_{i}.npy')
        # print(f'{dl_full=}')
        # print(f'{dl_partial=}')

        dl_full_list.append(dl_full)
        dl_partial_list.append(dl_partial)
        dl_no_beam_full_list.append(dl_no_beam_full)
        dl_no_beam_partial_list.append(dl_no_beam_partial)

    dl_full_arr = np.array(dl_full_list)
    dl_partial_arr = np.array(dl_partial_list)
    dl_no_beam_full_arr = np.array(dl_no_beam_full_list)
    dl_no_beam_partial_arr = np.array(dl_no_beam_partial_list)

    dl_full_mean = np.mean(dl_full_arr, axis=0)
    dl_partial_mean = np.mean(dl_partial_arr, axis=0)
    dl_no_beam_full_mean = np.mean(dl_no_beam_full_arr, axis=0)
    dl_no_beam_partial_mean = np.mean(dl_no_beam_partial_arr, axis=0)

    plt.figure(1)
    # plt.plot(l*(l+1)*cls_cmb[0,:lmax+1]/(2*np.pi), label='input')
    plt.plot(ell_arr, dl_true, label='input')
    plt.plot(ell_arr, dl_partial_mean, label='partial sky Namaster')
    plt.plot(ell_arr, dl_full_mean, label='full sky Namaster')
    plt.plot(ell_arr, dl_no_beam_partial_mean, label='partial sky Namaster no beam')
    plt.plot(ell_arr, dl_no_beam_full_mean, label='full sky Namaster no beam')
    plt.semilogy()

    plt.legend()
    plt.title('power spectrum')
    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell^{BB}$')
    plt.ylim(1e-5,1e5)

    plt.figure(2)
    plt.plot(ell_arr, np.abs(dl_partial_mean - dl_true)/dl_true, label='partial relative error')
    plt.plot(ell_arr, np.abs(dl_full_mean - dl_true)/dl_true, label='full relative error')
    plt.plot(ell_arr, np.abs(dl_no_beam_partial_mean - dl_true)/dl_true, label='partial relative error no beam')
    plt.plot(ell_arr, np.abs(dl_no_beam_full_mean - dl_true)/dl_true, label='full relative error no beam')
    plt.legend()
    # plt.semilogy()
    plt.ylim(0,1)
    plt.title('power spectrum')
    plt.xlabel('$\\ell$')
    plt.ylabel('abs(exp-true)/true')


    plt.show()

    # path_save = Path('/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20240926')
    # path_save.mkdir(exist_ok=True, parents=True)
    # plt.savefig(path_save / Path('mean.png'), dpi=300)

cpr_mode_mix()

