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

def gen_fg_cl():
    l = np.arange(8001)

    cl_TT = 10 / (l)**2.2
    cl_TT[0:2] = 0
    cl_TE = 2 / (l)**2.2
    cl_TE[0:2] = 0
    cl_EE = 1 / (l)**2.2
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

    np.save('./data/cl_fg/cl_fg_110.npy', np.asarray([cl_TT, cl_EE, cl_BB, cl_TE]))

# gen_fg_cl()


def test_fg():
    cls_fg = np.load('./data/cl_fg/cl_fg_86.npy')
    np.random.seed(seed=fg_seeds[rlz_idx])
    fg_rlz = hp.synfast(cls_fg, nside=nside, new=True)

    cls_rlz = hp.anafast(fg_rlz, lmax=lmax)

    path_cls = Path('./data/cl_rlz')
    path_cls.mkdir(exist_ok=True, parents=True)
    np.save(path_cls / Path(f'{rlz_idx}.npy'), cls_rlz)

    # plt.plot(l*(l+1)*cls_fg[2,:lmax+1]/(2*np.pi), label='input')
    # plt.plot(l*(l+1)*cls_rlz[2]/(2*np.pi), label='one realization')
    # plt.legend()
    # plt.show()

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
    cls_fg = np.load('./data/cl_fg/cl_fg_110.npy')
    np.random.seed(seed=fg_seeds[rlz_idx])
    fg_rlz = hp.synfast(cls_fg, nside=nside, new=True)
    return fg_rlz

def calc_dl_from_scalar_map(scalar_map, apo_mask, bin_dl, masked_on_input):
    scalar_field = nmt.NmtField(apo_mask, [scalar_map], masked_on_input=masked_on_input)
    dl = nmt.compute_full_master(scalar_field, scalar_field, bin_dl)
    return dl[0]

def calc_dl_from_pol_map(m_q, m_u, apo_mask, bin_dl, masked_on_input, purify_b):
    pol_field = nmt.NmtField(apo_mask, [m_q, m_u], masked_on_input=masked_on_input, purify_b=purify_b)
    dl = nmt.compute_full_master(pol_field, pol_field, bin_dl)
    return dl[3]

def test_mode_mix():
    # l_min_edges, l_max_edges = generate_bins(l_min_start=4, delta_l_min=20, l_max=lmax, fold=0.3)
    # bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=20, is_Dell=True) 
    bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=20, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    bin_mask = np.load('../../src/mask/north/BINMASKG1024.npy')
    apo_mask = nmt.mask_apodization(mask_in=bin_mask, aposize=5)

    m = get_fg()
    dl = calc_dl_from_pol_map(m_q=m[1]*bin_mask, m_u=m[2]*bin_mask, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    cl_no_mask = hp.anafast(m, lmax=lmax)[2]

    cls_fg = np.load('./data/cl_fg/cl_fg_10010.npy')
    dl_true = bin_dl.bin_cell(cls_fg[2,:lmax+1])
    dl_no_mask = bin_dl.bin_cell(cl_no_mask)
    print(f'{dl_true=}')

    path_dl_rlz = Path('./data/dls_fg_10010')
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

test_mode_mix()

def check_mode_mix():

    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=20, is_Dell=True)
    bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=20, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    cls_fg = np.load('./data/cl_fg/cl_fg_10010.npy')
    dl_true = bin_dl.bin_cell(cls_fg[2,:lmax+1])

    dl_full_list = []
    dl_partial_list = []
    for i in range(200):
        dl_full = np.load(f'./data/dls_fg_10010/full_{i}.npy')
        dl_partial = np.load(f'./data/dls_fg_10010/partial_{i}.npy')
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

    plt.legend()
    plt.title('power spectrum')
    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell^{BB}$')

    plt.figure(2)
    plt.plot(ell_arr, np.abs(dl_partial_mean - dl_true)/dl_true, label='partial relative error')
    plt.plot(ell_arr, np.abs(dl_full_mean - dl_true)/dl_true, label='full relative error')
    plt.legend()
    plt.title('power spectrum')
    plt.xlabel('$\\ell$')
    plt.ylabel('abs(exp-true)/true')


    plt.show()

    # path_save = Path('/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20240926')
    # path_save.mkdir(exist_ok=True, parents=True)
    # plt.savefig(path_save / Path('mean.png'), dpi=300)


# check_mode_mix()

def check_mode_mix_all():

    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=20, is_Dell=True)
    bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=20, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    cls_fg = np.load('./data/cl_fg/cl_fg_110.npy')
    cls_fg_1 = np.load('./data/cl_fg/cl_fg_1010.npy')
    cls_fg_2 = np.load('./data/cl_fg/cl_fg_10010.npy')
    l = np.arange(lmax+1)
    print(f'{l.shape=}')

    # plt.plot(l**2 * cls_fg[2,:lmax+1])
    # plt.plot(l**2 * cls_fg_1[2,:lmax+1])
    # plt.plot(l**2 * cls_fg_2[2,:lmax+1])
    # plt.show()

    dl_true = bin_dl.bin_cell(cls_fg[2,:lmax+1])

    dl_full_list = []
    dl_partial_list = []
    dl_partial_list_1 = []
    dl_partial_list_2 = []
    for i in range(200):
        dl_full = np.load(f'./data/dls_fg_110/full_{i}.npy')
        dl_partial = np.load(f'./data/dls_fg_110/partial_{i}.npy')
        dl_partial_1 = np.load(f'./data/dls_fg_1010/partial_{i}.npy')
        dl_partial_2 = np.load(f'./data/dls_fg_10010/partial_{i}.npy')
        print(f'{dl_full=}')
        print(f'{dl_partial=}')

        dl_full_list.append(dl_full)
        dl_partial_list.append(dl_partial)
        dl_partial_list_1.append(dl_partial_1)
        dl_partial_list_2.append(dl_partial_2)

    dl_full_arr = np.array(dl_full_list)
    dl_partial_arr = np.array(dl_partial_list)
    dl_partial_arr_1 = np.array(dl_partial_list_1)
    dl_partial_arr_2 = np.array(dl_partial_list_2)

    dl_full_mean = np.mean(dl_full_arr, axis=0)
    dl_partial_mean = np.mean(dl_partial_arr, axis=0)
    dl_partial_mean_1 = np.mean(dl_partial_arr_1, axis=0)
    dl_partial_mean_2 = np.mean(dl_partial_arr_2, axis=0)

    plt.figure(1)
    # plt.plot(l*(l+1)*cls_cmb[0,:lmax+1]/(2*np.pi), label='input')
    plt.plot(ell_arr, dl_true, label='input')
    plt.plot(ell_arr, dl_partial_mean, label='partial sky Namaster E/B=1/10')
    plt.plot(ell_arr, dl_partial_mean_1, label='partial sky Namaster E/B=1/1')
    plt.plot(ell_arr, dl_partial_mean_2, label='partial sky Namaster E/B=10/1')
    plt.plot(ell_arr, dl_full_mean, label='full sky Namaster')

    plt.legend()
    plt.title('power spectrum')
    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell^{BB}$')

    plt.figure(2)
    plt.plot(ell_arr, np.abs(dl_partial_mean - dl_true)/dl_true, label='partial relative error E/B=1/10')
    plt.plot(ell_arr, np.abs(dl_partial_mean_1 - dl_true)/dl_true, label='partial relative error E/B=1/1')
    plt.plot(ell_arr, np.abs(dl_partial_mean_2 - dl_true)/dl_true, label='partial relative error E/B=10/1')
    plt.plot(ell_arr, np.abs(dl_full_mean - dl_true)/dl_true, label='full relative error')
    plt.legend()
    plt.title('power spectrum')
    plt.xlabel('$\\ell$')
    plt.ylabel('abs(exp-true)/true')


    plt.show()

    # path_save = Path('/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20240926')
    # path_save.mkdir(exist_ok=True, parents=True)
    # plt.savefig(path_save / Path('mean.png'), dpi=300)


# check_mode_mix_all()

