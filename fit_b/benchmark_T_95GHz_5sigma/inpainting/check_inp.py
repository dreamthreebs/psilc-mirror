import numpy as np
import healpy as hp
import pandas as pd
import os,sys
import matplotlib.pyplot as plt

from pathlib import Path
config_dir = Path(__file__).parent.parent
print(f'{config_dir=}')
sys.path.insert(0, str(config_dir))
from config import freq, lmax, nside, beam

flux_idx = 2
rlz_idx = 0
df = pd.read_csv(f'../mask/{freq}.csv')
npix = hp.nside2npix(nside=nside)

def gen_map(lmax, freq, beam):
    # ps = np.load('../data/ps/ps.npy')

    nstd = np.load(f'../../../FGSim/NSTDNORTH/2048/{freq}.npy')
    # np.random.seed(seed=noise_seed[rlz_idx])
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    print(f"{np.std(noise[1])=}")

    # # cmb_iqu = np.load(f'../../fitdata/2048/CMB/215/{rlz_idx}.npy')
    # # cls = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')
    # cls = np.load('../../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    # np.random.seed(seed=cmb_seed[rlz_idx])
    # # cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=1999)
    # cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=lmax)

    # l = np.arange(lmax+1)
    # cls_out = hp.anafast(cmb_iqu, lmax=lmax)

    # cls_fg = gen_fg_cl()
    # np.random.seed(seed=fg_seed[rlz_idx])
    # fg_iqu = hp.synfast(cls_fg, nside=nside, fwhm=0, new=True, lmax=lmax)


    # m = noise + ps + cmb_iqu + fg_iqu
    m = noise
    # cn = noise + cmb_iqu

    # m = np.load('./1_8k.npy')
    # np.save('./1_6k_pcn.npy', m)
    # np.save('./1_6k_cn.npy', cn)
    return m, noise


def check_inp():

    input_pcfn = hp.read_map(f'./input/{rlz_idx}.fits')
    input_pcfn_1031 = hp.read_map(f'./input_1031/{rlz_idx}.fits')
    output_pcfn = hp.read_map(f'./output/{rlz_idx}.fits')
    output_pcfn_1031 = hp.read_map(f'./output_1031/{rlz_idx}.fits')

    for flux_idx in range(10):
        print(f'{flux_idx=}')
        lon = np.rad2deg(df.at[flux_idx, 'lon'])
        lat = np.rad2deg(df.at[flux_idx, 'lat'])

        hp.gnomview(input_pcfn, rot=[lon, lat, 0])
        hp.gnomview(input_pcfn_1031, rot=[lon, lat, 0])
        hp.gnomview(output_pcfn, rot=[lon, lat, 0])
        hp.gnomview(output_pcfn_1031, rot=[lon, lat, 0])
        plt.show()

def check_slope():
    for rlz_idx in range(200):
        slope = np.load(f'./eblc_slope/{rlz_idx}.npy')
        print(f'{slope=}')

def check_map_lmax():
    noise,_ = gen_map(lmax=lmax, freq=freq, beam=beam)
    cl = hp.anafast(noise, lmax=3*nside-1)
    plt.loglog(cl[2])
    plt.show()

def check_full_b():
    pcfn = hp.read_map('./test_1101_cf/0.fits')
    inp_m2 = hp.read_map('./test_1101_m2/0.fits')
    inp_m3 = hp.read_map('./test_1102_m3/0.fits')
    inp_m4 = hp.read_map('./test_1101_m4/0.fits')

    for flux_idx in range(10):
        print(f'{flux_idx=}')
        lon = np.rad2deg(df.at[flux_idx, 'lon'])
        lat = np.rad2deg(df.at[flux_idx, 'lat'])

        hp.gnomview(pcfn, rot=[lon, lat, 0], title='cf')
        hp.gnomview(inp_m2, rot=[lon, lat, 0], title='m2')
        hp.gnomview(inp_m3, rot=[lon, lat, 0], title='m3')
        hp.gnomview(inp_m4, rot=[lon, lat, 0], title='m4')
        plt.show()

def check_full_inp():

    pcfn = hp.read_map('./test_1101_cf/0.fits')
    # inp_m2 = hp.read_map('./test_1101_m2/0.fits')
    # inp_m3 = hp.read_map('./test_1103_m2/0.fits')
    # inp_m4 = hp.read_map('./test_1101_m3/0.fits')
    # inp_m5 = hp.read_map('./test_1103_m3/0.fits')
    # inp_m6 = hp.read_map('./test_1104_m3_lmax1k/0.fits')

    inp_m7 = hp.read_map('./test_1104_crt_cf/0.fits')
    inp_m8 = hp.read_map('./test_1104_cln_cf/0.fits')

    # inp_m6 = hp.read_map('./test_1104_m2/0.fits')
    # inp_m7 = hp.read_map('./test_1104_m3/0.fits')
    # inp_m8 = hp.read_map('./test_1104_cf/0.fits')

    for flux_idx in range(4,5):
        print(f'{flux_idx=}')
        lon = np.rad2deg(df.at[flux_idx, 'lon'])
        lat = np.rad2deg(df.at[flux_idx, 'lat'])

        hp.gnomview(pcfn, rot=[lon, lat, 0], title='cf')

        # hp.gnomview(inp_m2, rot=[lon, lat, 0], title='part m2')
        # hp.gnomview(inp_m3, rot=[lon, lat, 0], title='full m2')
        # hp.gnomview(inp_m4, rot=[lon, lat, 0], title='part m3')
        # hp.gnomview(inp_m5, rot=[lon, lat, 0], title='full m3')
        # hp.gnomview(inp_m6, rot=[lon, lat, 0], title='full m3 lmax 1k')

        hp.gnomview(inp_m7, rot=[lon, lat, 0], title='crt cf')
        hp.gnomview(inp_m8, rot=[lon, lat, 0], title='cln cf')

        # hp.gnomview(inp_m6, rot=[lon, lat, 0], title='T m2 ')
        # hp.gnomview(inp_m7, rot=[lon, lat, 0], title='T m3')
        # hp.gnomview(inp_m8, rot=[lon, lat, 0], title='input T')

        plt.show()

def check_1105():

    # cln_pcfn = hp.read_map('./test_1105_cln_pcfn/0.fits')
    # cln_n = hp.read_map('./test_1105_cln_n/0.fits')
    # inp_m2_pcfn = hp.read_map('./output_1105_pcfn_m2/0.fits')
    # inp_m2_n = hp.read_map('./output_1105_n_m2/0.fits')
    inp_m2_pcfn_l6k = hp.read_map('./output_1105_pcfn_m2_l6k/0.fits')
    inp_m2_n_l6k = hp.read_map('./output_1105_n_m2_l6k/0.fits')
    inp_m3_pcfn_l6k = hp.read_map('./output_1105_pcfn_m3_l6k/0.fits')
    inp_m3_n_l6k = hp.read_map('./output_1105_n_m3_l6k/0.fits')

    inp = hp.read_map('./test_1105_cln_pcfn/0.fits')
    edge_pcfn = hp.read_map('./test_1105_cln_pcfn_edge/0.fits')
    inp_edge_pcfn_m3 = hp.read_map('./output_1105_pcfn_m3_l6k_edge/0.fits')
    inp_edge_pcfn_m2 = hp.read_map('./output_1105_pcfn_m2_l6k_edge/0.fits')
    inp_edge_n_m2 = hp.read_map('./output_1105_n_m2_l6k_edge/0.fits')
    inp_edge_n_m3 = hp.read_map('./output_1105_n_m3_l6k_edge/0.fits')

    for flux_idx in range(5,6):
        print(f'{flux_idx=}')
        lon = np.rad2deg(df.at[flux_idx, 'lon'])
        lat = np.rad2deg(df.at[flux_idx, 'lat'])

        # hp.gnomview(cln_pcfn, rot=[lon, lat, 0], title='cln_pcfn')
        # hp.gnomview(cln_n, rot=[lon, lat, 0], title='cln_n')
        # hp.gnomview(inp_m2_pcfn, rot=[lon, lat, 0], title='inp m2 pcfn')
        # hp.gnomview(inp_m2_n, rot=[lon, lat, 0], title='inp m2 n')
        hp.gnomview(inp, rot=[lon, lat, 0], title='inp before')
        hp.gnomview(inp_m2_pcfn_l6k, rot=[lon, lat, 0], title='inp m2 l6k pcfn')
        hp.gnomview(inp_m2_n_l6k, rot=[lon, lat, 0], title='inp m2 l6k n')
        hp.gnomview(inp_m3_pcfn_l6k, rot=[lon, lat, 0], title='inp m3 l6k pcfn')
        hp.gnomview(inp_m3_n_l6k, rot=[lon, lat, 0], title='inp m3 l6k n')

        hp.gnomview(edge_pcfn, rot=[lon, lat, 0], title='edge_pcfn')
        hp.gnomview(inp_edge_pcfn_m2, rot=[lon, lat, 0], title='inp_edge_pcfn_m2')
        hp.gnomview(inp_edge_n_m2, rot=[lon, lat, 0], title='inp_edge_n_m2')
        hp.gnomview(inp_edge_n_m3, rot=[lon, lat, 0], title='inp_edge_n_m3')
        hp.gnomview(inp_edge_pcfn_m3, rot=[lon, lat, 0], title='inp_edge_pcfn_m3')
        plt.show()

def check_1106():

    # pcfn_1k = hp.read_map('./test_1106_pcfn/0.fits')
    # n_1k = hp.read_map('./test_1106_n/0.fits')
    # cut_pcfn_1k = hp.read_map('./test_1106_pcfn_cutqu/0.fits')
    # inp_pcfn_1k_m3 = hp.read_map('./output_1106_pcfn_m3_l1k/0.fits')
    # inp_pcfn_1k_m2 = hp.read_map('./output_1106_/0.fits')
    # inp_pcfn_1k_m2 = hp.read_map('./output_1107_/0.fits')
    # cut_n_1k = hp.read_map('./test_1106_n_cutqu/0.fits')
    # inp_n_1k = hp.read_map('./output_1106_n_m3_l1k/0.fits')

    mask = hp.read_map(f'./mask/mask_1dot8.fits')

    edge_pcfn_1k = hp.read_map('./test_1107_pcfn_edge/0.fits')
    # edge_n_1k = hp.read_map('./test_1107_n_edge/0.fits')
    edge_cfn_1k = hp.read_map('./test_1107_cfn_edge/0.fits')

    hole_pcfn_1k = hp.read_map('./test_1107_pcfn_hole/0.fits')
    # hole_cfn_1k = hp.read_map('./test_1107_cfn_hole/0.fits')
    # hole_n_1k = hp.read_map('./test_1107_n_hole/0.fits')


    vmin = -2
    vmax = 2

    for flux_idx in range(5,6):
        print(f'{flux_idx=}')
        lon = np.rad2deg(df.at[flux_idx, 'lon'])
        lat = np.rad2deg(df.at[flux_idx, 'lat'])

        # hp.gnomview(pcfn_1k, rot=[lon, lat, 0], title='pcfn 1k', min=vmin, max=vmax)
        # hp.gnomview(n_1k, rot=[lon, lat, 0], title='n 1k', min=vmin, max=vmax)
        # hp.gnomview(cut_n_1k, rot=[lon, lat, 0], title='cut n 1k', min=vmin, max=vmax)
        # hp.gnomview(cut_pcfn_1k, rot=[lon, lat, 0], title='cut pcfn 1k', min=vmin, max=vmax)
        # hp.gnomview(inp_pcfn_1k, rot=[lon, lat, 0], title='inp pcfn', min=vmin, max=vmax)
        # hp.gnomview(inp_n_1k, rot=[lon, lat, 0], title='inp n', min=vmin, max=vmax)

        # hp.gnomview(edge_n_1k, rot=[lon, lat, 0], title='edge n', min=vmin, max=vmax)
        hp.gnomview(edge_pcfn_1k, rot=[lon, lat, 0], title='point source + cmb + fg + noise', min=vmin, max=vmax, xsize=300)
        # hp.gnomview(edge_cfn_1k, rot=[lon, lat, 0], title='edge cfn', min=vmin, max=vmax)

        # hp.gnomview(hole_n_1k, rot=[lon, lat, 0], title='hole n', min=vmin, max=vmax)
        hp.gnomview(hole_pcfn_1k, rot=[lon, lat, 0], title='with hole(1.5 beam size disc) point source + cmb + fg + noise', min=vmin, max=vmax, xsize=300)
        # hp.gnomview(inp_pcfn_1k_m2, rot=[lon, lat, 0], title='inp pcfn 1k m2', min=vmin, max=vmax)
        # hp.gnomview(inp_pcfn_1k_m3, rot=[lon, lat, 0], title='inp pcfn 1k m3', min=vmin, max=vmax)

        hp.gnomview((hole_pcfn_1k - edge_cfn_1k)*mask, rot=[lon, lat, 0], title='with hole EB leakage(1.8 beam size disc)', xsize=300, min=-1.5, max=1.5)
        hp.gnomview((edge_pcfn_1k - edge_cfn_1k)*mask, rot=[lon, lat, 0], title='no hole point source contribution(1.8 beam size disc)', xsize=300, min=-1.5, max=1.5)
        # hp.gnomview(hole_cfn_1k - edge_cfn_1k, rot=[lon, lat, 0], title='res', xsize=300)


        plt.show()

def check_1108():


    mask = hp.read_map(f'./mask/mask_1dot8.fits')

    edge_pcfn_1k = hp.read_map('./test_1108_pcfn_edge/0.fits')
    edge_n_1k = hp.read_map('./test_1108_n_edge/0.fits')
    edge_inp_1k_m2_pcfn = hp.read_map('./output_1108_pcfn_m2_l1k_edge/0.fits')
    edge_inp_1k_m2_n = hp.read_map('./output_1108_n_m2_l1k_edge/0.fits')
    edge_inp_1k_m3_pcfn = hp.read_map('./output_1108_pcfn_m3_l1k_edge/0.fits')
    edge_inp_1k_m3_n = hp.read_map('./output_1108_n_m3_l1k_edge/0.fits')

    hole_pcfn_1k = hp.read_map('./test_1108_pcfn_hole/0.fits')
    hole_n_1k = hp.read_map('./test_1108_n_hole/0.fits')
    hole_inp_1k_m2_pcfn = hp.read_map('./output_1108_pcfn_m2_l1k_hole/0.fits')
    hole_inp_1k_m2_n = hp.read_map('./output_1108_n_m2_l1k_hole/0.fits')
    hole_inp_1k_m3_pcfn = hp.read_map('./output_1108_pcfn_m3_l1k_hole/0.fits')
    hole_inp_1k_m3_n = hp.read_map('./output_1108_n_m3_l1k_hole/0.fits')

    vmin = -1.5
    vmax = 1.5

    for flux_idx in range(5,6):
        print(f'{flux_idx=}')
        lon = np.rad2deg(df.at[flux_idx, 'lon'])
        lat = np.rad2deg(df.at[flux_idx, 'lat'])

        # hp.gnomview(edge_pcfn_1k, rot=[lon, lat, 0], title='no hole point source + cmb + fg + noise', min=vmin, max=vmax)
        # hp.gnomview(edge_n_1k, rot=[lon, lat, 0], title='no hole noise', min=vmin, max=vmax)
        # hp.gnomview(edge_inp_1k_m2_pcfn, rot=[lon, lat, 0], title='no hole inp m2 point source + cmb + fg + noise', min=vmin, max=vmax)
        # hp.gnomview(edge_inp_1k_m2_n, rot=[lon, lat, 0], title='no hole inp m2 noise', min=vmin, max=vmax)
        # hp.gnomview(edge_inp_1k_m3_pcfn, rot=[lon, lat, 0], title='no hole inp m3 point source + cmb + fg + noise', min=vmin, max=vmax)
        # hp.gnomview(edge_inp_1k_m3_n, rot=[lon, lat, 0], title='no hole inp m3 noise', min=vmin, max=vmax)

        hp.gnomview(hole_pcfn_1k, rot=[lon, lat, 0], title='with hole point source + cmb + fg + noise', min=vmin, max=vmax)
        hp.gnomview(hole_n_1k, rot=[lon, lat, 0], title='with hole noise', min=vmin, max=vmax)
        hp.gnomview(hole_inp_1k_m2_pcfn, rot=[lon, lat, 0], title='with hole inp m2 point source + cmb + fg + noise', min=vmin, max=vmax)
        hp.gnomview(hole_inp_1k_m2_n, rot=[lon, lat, 0], title='with hole inp m2 noise', min=vmin, max=vmax)
        hp.gnomview(hole_inp_1k_m3_pcfn, rot=[lon, lat, 0], title='with hole inp m3 point source + cmb + fg + noise', min=vmin, max=vmax)
        hp.gnomview(hole_inp_1k_m3_n, rot=[lon, lat, 0], title='with hole inp m3 noise', min=vmin, max=vmax)



        plt.show()




def check_cl_cf():
    cf = hp.read_map('test_1101_cf/0.fits')
    cl = hp.anafast(cf, lmax=2000)
    plt.plot(cl)
    plt.loglog()
    plt.show()


# check_slope()
# check_inp()
# check_map_lmax()
# check_full_b()
# check_cl_cf()
# check_full_inp()
# check_1105()
# check_1106()
check_1108()

