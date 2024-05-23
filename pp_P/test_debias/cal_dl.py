import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt

from pathlib import Path

beam = 63
lmax = 400
nside = 256

rlz_idx = 0
bin_mask_2048 = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
apo_mask_2048 = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')

bin_mask = hp.ud_grade(bin_mask_2048, nside_out=nside)
bin_mask[bin_mask < 1] = 0
apo_mask = hp.ud_grade(apo_mask_2048, nside_out=nside)

# m_c = np.load(f'./cmb/B/{rlz_idx}.npy')

# m_cn = np.load(f'./cn/B/{rlz_idx}.npy')
# m_n = np.load(f'./noise/B/{rlz_idx}.npy')

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

def calc_dl_from_scalar_map(scalar_map, bl, apo_mask, bin_dl, masked_on_input):
    scalar_field = nmt.NmtField(apo_mask, [scalar_map], beam=bl, masked_on_input=masked_on_input)
    dl = nmt.compute_full_master(scalar_field, scalar_field, bin_dl)
    return dl[0]

def check_mask_effect():
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax, fold=0.2)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    dl_c_b = calc_dl_from_scalar_map(m_c, bl, apo_mask, bin_dl, masked_on_input=False)
    dl_c_b_bin_mask = calc_dl_from_scalar_map(m_c*bin_mask, bl, apo_mask, bin_dl, masked_on_input=False)

    plt.plot(ell_arr, dl_c_b, label='c b', marker='o')
    plt.plot(ell_arr, dl_c_b_bin_mask, label='c b with mask', marker='o')
    plt.semilogy()
    plt.show()


def check_realization_effect():
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax, fold=0.2)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    dl_cn_b_list = []
    dl_c_b_list = []
    dl_n_b_list = []
    for rlz_idx in range(1000):
        print(f'{rlz_idx=}')
        m_cn = np.load(f'./cn/B/{rlz_idx}.npy')
        m_c = np.load(f'./cmb/B/{rlz_idx}.npy')
        m_n = np.load(f'./noise/B/{rlz_idx}.npy')
        dl_cn_b = calc_dl_from_scalar_map(m_cn, bl, apo_mask, bin_dl, masked_on_input=False)
        dl_c_b = calc_dl_from_scalar_map(m_c, bl, apo_mask, bin_dl, masked_on_input=False)
        dl_n_b = calc_dl_from_scalar_map(m_n, bl, apo_mask, bin_dl, masked_on_input=False)

        dl_cn_b_list.append(dl_cn_b)
        dl_c_b_list.append(dl_c_b)
        dl_n_b_list.append(dl_n_b)

    dl_cn_b_arr = np.array(dl_cn_b_list)
    dl_c_b_arr = np.array(dl_c_b_list)
    dl_n_b_arr = np.array(dl_n_b_list)

    path_dl = Path(f'./dl_result')
    path_dl.mkdir(exist_ok=True, parents=True)

    np.save(path_dl / Path(f'dl_cn_b.npy'), dl_cn_b_arr)
    np.save(path_dl / Path(f'dl_c_b.npy'), dl_c_b_arr)
    np.save(path_dl / Path(f'dl_n_b.npy'), dl_n_b_arr)

def check_dl():

    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax, fold=0.2)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    dl_cn = np.load('./dl_result/dl_cn_b.npy')
    print(f'{dl_cn.shape=}')
    dl_c = np.load('./dl_result/dl_c_b.npy')
    dl_n = np.load('./dl_result/dl_n_b.npy')

    dl_cn_mean = np.mean(dl_cn, axis=0)
    dl_c_mean = np.mean(dl_c, axis=0)
    dl_n_mean = np.mean(dl_n, axis=0)


    dl_cn_100 = dl_cn[:100,:]
    print(f'{dl_cn_100.shape=}')
    dl_c_100 = dl_c[:100,:]
    dl_n_100 = dl_n[:100,:]
    dl_n_100_1 = dl_n[900:1000,:]

    dl_cn_mean_100 = np.mean(dl_cn_100, axis=0)
    dl_c_mean_100 = np.mean(dl_c_100, axis=0)
    dl_n_mean_100 = np.mean(dl_n_100, axis=0)
    dl_n_mean_100_1 = np.mean(dl_n_100_1, axis=0)

    plt.plot(ell_arr, dl_n_mean, label='n_1000')
    plt.plot(ell_arr, dl_cn_mean_100 - dl_n_mean_100, label='cn_100 - n_100_ori')
    plt.plot(ell_arr, dl_cn_mean_100 - dl_n_mean_100_1, label='cn_100 - n_100_other')
    plt.plot(ell_arr, dl_cn_mean_100 - dl_n_mean, label='cn_100 - n_1000')
    plt.plot(ell_arr, dl_cn_mean - dl_n_mean, label='cn_1000 - n_1000')
    plt.plot(ell_arr, dl_c_mean, label='c_1000')
    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell^{BB} [\\mu K^2]$')
    plt.semilogy()
    plt.legend()
    plt.show()


if __name__ == "__main__":

    # check_mask_effect()
    # check_realization_effect()
    check_dl()

