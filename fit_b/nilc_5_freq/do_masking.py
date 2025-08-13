import numpy as np
import healpy as hp
import pandas as pd
import time
import matplotlib.pyplot as plt
import pymaster as nmt
from pathlib import Path

rlz_idx = 0
freq_list = [30, 95, 155, 215, 270]
beam_list = [67, 30, 17, 11, 9]
beam_base = 17 #arcmin
lmax = 1500
nside = 2048


def calc_lmax():
    for freq, beam in zip(freq_list, beam_list):
        lmax = int(2 * np.pi / np.deg2rad(beam) * 60) + 1
        print(f'{freq=}, {beam=}, {lmax=}')

def collect_diff_freq_maps():
    map_list = []
    for freq in freq_list:
        m = np.load(f'../{freq}GHz/fit_res/sm/noise/n/0.npy')
        # m = np.load(f'../30GHz/fit_res/sm/noise/n/0.npy')
        map_list.append(m)
    map_arr = np.asarray(map_list)
    print(f'{map_arr.shape=}')
    np.save('./test/n.npy', map_arr)

def check_do_pcfn():
    mask = np.load(f'../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_3APO_5.npy')
    fsky = np.sum(mask) / np.size(mask)
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam_base)/60, lmax=lmax, pol=True)[:,2]
    df_ps = pd.read_csv(f'../95GHz/mask/95_after_filter.csv')
    cln_pcfn = np.load(f'./data2/std/pcfn/0.npy')
    cln_cfn = np.load(f'./data2/std/cfn/0.npy')
    cln_rmv = np.load(f'./data2/std/rmv/0.npy')
    cln_inp = np.load(f'./data2/std/inp/0.npy')
    # cln_n = np.load(f'./data/mean/n/0.npy')
    hp.orthview(cln_pcfn, rot=[100,50,0], title='pcfn')
    hp.orthview(cln_cfn, rot=[100,50,0], title='cfn')
    hp.orthview(cln_rmv, rot=[100,50,0], title='rmv')
    hp.orthview(cln_inp, rot=[100,50,0], title='inp')
    plt.show()

    for flux_idx in np.arange(len(df_ps)):
        lon = np.rad2deg(df_ps.at[flux_idx, 'lon'])
        lat = np.rad2deg(df_ps.at[flux_idx, 'lat'])
        hp.gnomview(cln_pcfn, rot=[lon, lat, 0], title='pcfn')
        hp.gnomview(cln_cfn, rot=[lon, lat, 0], title='cfn')
        hp.gnomview(cln_rmv, rot=[lon, lat, 0], title='rmv')
        hp.gnomview(cln_inp, rot=[lon, lat, 0], title='inp')
        # hp.gnomview(cln_cf, rot=[lon, lat, 0], title='cf')
        plt.show()

    # cl_pcfn = hp.anafast(cln_pcfn, lmax=lmax)
    # cl_n = hp.anafast(cln_n, lmax=lmax)
    # cl_fid = gen_fiducial_cmb()
    # l = np.arange(np.size(cl_pcfn))
    # plt.loglog(l, l*(l+1)*(cl_pcfn-cl_n)/bl**2/fsky, label='debias pcfn')
    # plt.loglog(l, l*(l+1)*cl_n/bl**2/fsky, label='noise')
    # plt.loglog(l, l*(l+1)*cl_fid/bl**2, label='fiducial')
    # plt.legend()
    # plt.show()

def calc_dl_from_scalar_map(scalar_map, apo_mask, bin_dl, masked_on_input):
    scalar_field = nmt.NmtField(apo_mask, [scalar_map], masked_on_input=masked_on_input, lmax=lmax, lmax_mask=lmax)
    dl = nmt.compute_full_master(scalar_field, scalar_field, bin_dl)
    return dl[0]

def calc_dl_from_scalar_map_bl(scalar_map, apo_mask, bl, bin_dl, masked_on_input):
    scalar_field = nmt.NmtField(apo_mask, [scalar_map], beam=bl, masked_on_input=masked_on_input, lmax=lmax, lmax_mask=lmax)
    dl = nmt.compute_full_master(scalar_field, scalar_field, bin_dl)
    return dl[0]

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

def gen_apodized_ps_mask():
    ori_mask = np.load(f"../../psfit/fitv4/fit_res/2048/ps_mask/new_mask/apo_C1_3_apo_3_apo_3.npy")
    freq = 30
    beam = beam_base
    beam = 67
    # df = pd.read_csv(f'../{freq}GHz/mask/{freq}_after_filter.csv')
    df = pd.read_csv(f'./concat_ps.csv')
    mask = np.ones(hp.nside2npix(nside))
    for flux_idx in range(len(df)):
        print(f'{flux_idx=}')
        if flux_idx > 40:
            break
        lon = np.rad2deg(df.at[flux_idx, 'lon'])
        lat = np.rad2deg(df.at[flux_idx, 'lat'])

        ctr_vec = hp.ang2vec(theta=lon, phi=lat, lonlat=True)
        ipix_mask = hp.query_disc(nside=nside, vec=ctr_vec, radius=2.5 * np.deg2rad(beam) / 60)
        mask[ipix_mask] = 0

        # fig_size=200
        # # hp.gnomview(ori_mask, rot=[lon, lat, 0], title='before mask', xsize=fig_size)
        # hp.gnomview(mask, rot=[lon, lat, 0], title='after mask', xsize=fig_size)
        # plt.show()

    # apo_mask = nmt.mask_apodization(mask_in=mask, aposize=1)
    # hp.orthview(apo_mask * ori_mask, rot=[100,50, 0], title='mask', xsize=2000)
    # plt.show()

    path_mask = Path('./ps_mask')
    path_mask.mkdir(exist_ok=True, parents=True)
    # np.save(f'./ps_mask/{freq}GHz_1deg.npy', apo_mask * ori_mask)
    # np.save(f'./ps_mask/{freq}GHz.npy', mask * ori_mask)
    np.save(f'./ps_mask/union_40.npy', mask * ori_mask)


def calc_cl_with_mask():
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam_base)/60, lmax=lmax, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=42, delta_l_min=40, l_max=lmax+1, fold=0.1, l_threshold=400)
    # delta_ell = 30
    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=40, is_Dell=True)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    freq = 30
    # mask_b = np.load(f"./ps_mask/{freq}GHz.npy")
    mask_b = np.load(f"./ps_mask/new_union_apo.npy")
    # mask_b = np.load(f"./ps_mask/new_{freq}GHz_3deg.npy")

    m_std = np.load(f'./data2/std/pcfn/{rlz_idx}.npy')
    m_n = np.load(f'./data2/std/n_pcfn/{rlz_idx}.npy')
    dl_std = calc_dl_from_scalar_map_bl(scalar_map=m_std, apo_mask=mask_b, bl=bl, bin_dl=bin_dl, masked_on_input=False)
    dl_n = calc_dl_from_scalar_map_bl(scalar_map=m_n, apo_mask=mask_b, bl=bl, bin_dl=bin_dl, masked_on_input=False)

    path_dl_std = Path(f'./dl_res4/mask')
    path_dl_std.mkdir(parents=True, exist_ok=True)

    # np.save(path_dl_std / Path(f'pcfn_{freq}_67_{rlz_idx}.npy'), dl_std)
    # np.save(path_dl_std / Path(f'n_{freq}_67_{rlz_idx}.npy'), dl_n)

    np.save(path_dl_std / Path(f'pcfn_union_new_apo_{rlz_idx}.npy'), dl_std)
    np.save(path_dl_std / Path(f'n_union_new_apo_{rlz_idx}.npy'), dl_n)



# calc theoretical bandpower!!!

def calc_th_bpw():
    # do not add beam if you are calculating covariance
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam_base)/60, lmax=lmax, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=42, delta_l_min=40, l_max=lmax+1, fold=0.1, l_threshold=400)
    # delta_ell = 30
    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=40, is_Dell=True)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    # apo_mask  = np.load(f"../../psfit/fitv4/fit_res/2048/ps_mask/new_mask/apo_C1_3_apo_3_apo_3.npy")
    apo_mask  = np.load(f"./ps_mask/new_union_apo.npy")
    # apo_mask  = np.load(f"./ps_mask/union.npy")
    hp.orthview(apo_mask, rot=[100,50, 0], title='mask', xsize=2000)
    plt.show()
    fsky = np.mean(apo_mask**2)

    f = nmt.NmtField(apo_mask, [np.ones_like(apo_mask)], masked_on_input=False, lmax=lmax, lmax_mask=lmax)
    w = nmt.NmtWorkspace.from_fields(f, f, bins=bin_dl)

    cl_cmb = np.load('/afs/ihep.ac.cn/users/w/wangyiming25/work/dc2/psilc/src/cmbsim/cmbdata/cmbcl_8k.npy').T
    # cl_cmb = np.load('/afs/ihep.ac.cn/users/w/wangyiming25/work/dc2/psilc/src/cmbsim/cmbdata/cmbcl_r_1.npy').T
    print(f'{cl_cmb.shape=}')
    cl_true = cl_cmb[2,:lmax+1]
    ells = np.arange(len(cl_true))
    # cl_std = np.sqrt(2/(2*ells+1)) * cl_true

    # b = bin_dl
    # ls = np.arange(lmax+1)
    # n_bins = b.get_n_bands()
    # Fls = np.zeros([n_bins, lmax+1])
    # for i in range(n_bins):
    #     Fls[i, b.get_ell_list(i)] = b.get_weight_list(i)
    # # plot bin weights
    # plt.figure(figsize=(7, 5))
    # plt.title('Bin weights')
    # for i, fl in enumerate(Fls):
    #     plt.plot(ls, fl, 'k-', alpha=(i+1)/n_bins)
    # plt.xlabel(r'$\ell$', fontsize=15)
    # plt.show()
    # # Extract the mode-coupling matrix
    # mcm = w.get_coupling_matrix()
    # # Extract bandpower window functions
    # Bbl = w.get_bandpower_windows().squeeze()
    # print(f"{Bbl.shape=}")
    # # Plot MCM
    # plt.figure(figsize=(5, 5))
    # plt.title('Mode-coupling matrix')
    # plt.imshow(mcm, cmap='bone')
    # plt.ylabel("$\\ell$", fontsize=15)
    # plt.xlabel("$\\ell'$", fontsize=15)
    # plt.colorbar()
    # plt.show()
    # # plot mcm at fixed ell
    # plt.figure(figsize=(7, 5))
    # for ll in [10, 50, 100, 150, 200]:
    #     plt.plot(ls, mcm[ll]/fsky, label=f'$\\ell={ll}$')
    # plt.xlabel("$\\ell'$", fontsize=15)
    # plt.ylabel("$M_{\\ell \\ell'}/f_{\\rm sky}$", fontsize=15)
    # plt.legend(fontsize=12)
    # plt.xlim([0, 300]) 
    # plt.show()
    # # plot bin vs bandpower windows
    # plt.figure(figsize=(7, 5))
    # plt.title('Bins vs. bandpower windows')
    # for i, fl in enumerate(Fls):
    #     plt.plot(ls, fl, 'k-', alpha=(i+1)/n_bins)
    #     plt.plot(ls, Bbl[i], 'r-', alpha=(i+1)/n_bins)
    # plt.xlabel(r'$\ell$', fontsize=15);
    # plt.show()

    # calculate theoretical bandpower
    dl_th = w.decouple_cell(w.couple_cell([cl_true]))[0]

    # # calculate the theoretical covariance
    # cw = nmt.NmtCovarianceWorkspace.from_fields(f, f, f, f)
    # cov_00_00 = nmt.gaussian_covariance(cw,
    #                                   0, 0, 0, 0,  # Spins of the 4 fields
    #                                   [cl_true],  # TT
    #                                   [cl_true],  # TT
    #                                   [cl_true],  # TT
    #                                   [cl_true],  # TT
    #                                   w)
    # dl_th_std = np.sqrt(np.diag(cov_00_00))


    path_dl_qu_mask = Path(f'./dl_res4/mask')
    path_dl_qu_mask.mkdir(parents=True, exist_ok=True)
    # np.save(path_dl_qu_mask/ Path(f'th_r_95_2deg_{rlz_idx}.npy'), dl_th)
    # np.save(path_dl_qu_mask/ Path(f'th_30_67_{rlz_idx}.npy'), dl_th)
    np.save(path_dl_qu_mask/ Path(f'th_union_new_apo_{rlz_idx}.npy'), dl_th)
    # np.save(path_dl_qu_mask/ Path(f'th_std_apo_new_{rlz_idx}.npy'), dl_th_std)


# test on bias from mask
def test_gen_mask():
    ori_mask = np.load(f"../../psfit/fitv4/fit_res/2048/ps_mask/new_mask/BIN_C1_3_C1_3.npy")

    freq = 95
    beam = beam_base
    df = pd.read_csv(f'../{freq}GHz/mask/{freq}_after_filter.csv')
    mask = np.ones(hp.nside2npix(nside))
    for flux_idx in range(len(df)):
        print(f'{flux_idx=}')
        lon = np.rad2deg(df.at[flux_idx, 'lon'])
        lat = np.rad2deg(df.at[flux_idx, 'lat'])

        ctr_vec = hp.ang2vec(theta=lon, phi=lat, lonlat=True)
        ipix_mask = hp.query_disc(nside=nside, vec=ctr_vec, radius=2.5 * np.deg2rad(beam) / 60)
        mask[ipix_mask] = 0

        # fig_size=200
        # # hp.gnomview(ori_mask, rot=[lon, lat, 0], title='before mask', xsize=fig_size)
        # hp.gnomview(mask, rot=[lon, lat, 0], title='after mask', xsize=fig_size)
        # plt.show()

    apo_mask = nmt.mask_apodization(mask_in=mask*ori_mask, aposize=3)
    hp.orthview(apo_mask, rot=[100,50, 0], title='mask', xsize=2000)
    plt.show()

    path_mask = Path('./ps_mask')
    path_mask.mkdir(exist_ok=True, parents=True)
    np.save(f'./ps_mask/new_{freq}GHz_3deg.npy', apo_mask * ori_mask)

def test_mask():
    mask_1 = np.load('./ps_mask/union.npy')
    # mask_2 = np.load('./ps_mask/new_95GHz_3deg.npy')
    hp.orthview(mask_1, rot=[100,50,0], half_sky=True)
    # hp.orthview(mask_2, rot=[100,50,0], half_sky=True)
    plt.show()

def check_cmb_bias_from_mask():
    pass


# concat the filtered point sources
def concat_df():
    df1 = pd.read_csv('../30GHz/mask/30_after_filter.csv')
    df2 = pd.read_csv('../95GHz/mask/95_after_filter.csv')
    df3 = pd.read_csv('../155GHz/mask/155_after_filter.csv')
    df4 = pd.read_csv('../215GHz/mask/215_after_filter.csv')
    df5 = pd.read_csv('../270GHz/mask/270_after_filter.csv')
    dfs = [df1, df2, df3, df4, df5]
    df_union = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="index", keep="first")

    df_union.to_csv("./concat_ps.csv", index=False)

def gen_union_mask():

    # mask = np.ones(hp.nside2npix(nside=nside))
    # for freq in freq_list:
    #     mask_freq = hp.read_map(f'../{freq}GHz/inpainting/new_mask/for_union.fits')
    #     hp.orthview(mask_freq, rot=[100,50,0], half_sky=True)
    #     plt.show()
    #     mask = mask * mask_freq

    # hp.orthview(mask, rot=[100,50,0], half_sky=True)
    # plt.show()

    # np.save('./ps_mask/new_union.npy', mask)

    mask = np.load('./ps_mask/new_union.npy')
    ori_mask = np.load(f"../../psfit/fitv4/fit_res/2048/ps_mask/new_mask/apo_C1_3_apo_3_apo_3.npy")

    hp.orthview(mask * ori_mask, rot=[100,50,0], half_sky=True)
    plt.show()


    np.save('./ps_mask/new_union_apo.npy', mask * ori_mask)



if __name__ == "__main__":
    # gen_apodized_ps_mask()
    # calc_cl_with_mask()

    calc_th_bpw()

    # test_gen_mask()
    # test_mask()

    # concat_df()
    # gen_union_mask()


