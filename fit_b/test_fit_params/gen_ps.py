import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from pathlib import Path

def gen_fg_cl(freq):
    cl_fg = np.load(f'../{freq}GHz/data/debeam_full_b/cl_fg.npy')
    Cl_TT = cl_fg[0]
    Cl_EE = cl_fg[1]
    Cl_BB = cl_fg[2]
    Cl_TE = np.zeros_like(Cl_TT)
    return np.array([Cl_TT, Cl_EE, Cl_BB, Cl_TE])

def gen_cmb_cl(beam, lmax):
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=10000, pol=True)
    print(f'{bl[0:10,0]=}')
    print(f'{bl[0:10,1]=}')
    print(f'{bl[0:10,2]=}')
    print(f'{bl[0:10,3]=}')
    # cl = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')
    cl = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    print(f'{cl.shape=}')

    Cl_TT = cl[0:lmax+1,0] * bl[0:lmax+1,0]**2
    Cl_EE = cl[0:lmax+1,1] * bl[0:lmax+1,1]**2
    Cl_BB = cl[0:lmax+1,2] * bl[0:lmax+1,2]**2
    Cl_TE = cl[0:lmax+1,3] * bl[0:lmax+1,3]**2
    return np.asarray([Cl_TT, Cl_EE, Cl_BB, Cl_TE])


def mJy_to_uKCMB(intensity_mJy, frequency_GHz):
    # Constants
    c = 2.99792458e8  # Speed of light in m/s
    h = 6.62607015e-34  # Planck constant in J*s
    k = 1.380649e-23  # Boltzmann constant in J/K
    T_CMB = 2.725  # CMB temperature in Kelvin

    frequency_Hz = frequency_GHz * 1e9 # Convert frequency to Hz from GHz

    x = (h * frequency_Hz) / (k * T_CMB) # Calculate x = h*nu/(k*T)

    # Calculate the derivative of the Planck function with respect to temperature, dB/dT
    dBdT = (2.0 * h * frequency_Hz**3 / c**2 / T_CMB) * (x * np.exp(x) / (np.exp(x) - 1)**2)
    intensity_Jy = intensity_mJy * 1e-3 # Convert intensity from mJy to Jy
    intensity_W_m2_sr_Hz = intensity_Jy * 1e-26 # Convert Jy/sr to W/m^2/sr/Hz
    uK_CMB = intensity_W_m2_sr_Hz / dBdT * 1e6 # Convert to uK_CMB, taking the inverse of dB/dT
    return uK_CMB

def gen_ps(freq, beam, nside, distance_factor, rlz_idx=0, qflux0=6000, uflux0=4000, qflux1=4000, uflux1=-6000):
    # basic parameters
    npix = hp.nside2npix(nside=nside)
    nside2pixarea_factor = hp.nside2pixarea(nside=nside)
    distance = distance_factor * beam / 60

    # empty map
    delta_t_map = np.zeros((npix,))
    delta_q_map = np.zeros((npix,))
    delta_u_map = np.zeros((npix,))
    print(f'{delta_t_map.shape=}')

    path_ps = Path('./data/ps')
    path_ps.mkdir(exist_ok=True, parents=True)

    # generate map
    for flux_idx in range(2):
        print(f'{flux_idx=}')
        if flux_idx == 0:
            lon = 0
            lat = 0
            qflux = qflux0
            uflux = uflux0

        elif flux_idx == 1:
            lon = distance
            lat = 0
            qflux = qflux1
            uflux = uflux1


        print(f'{lon=}, {lat=}, {qflux=}, {uflux=}')

        ps_pix_idx = hp.ang2pix(nside=nside, theta=lon, phi=lat, lonlat=True)
        pix_lon, pix_lat = hp.pix2ang(ipix=ps_pix_idx, nside=nside, lonlat=True)
        print(f'{ps_pix_idx=}')

        if rlz_idx == 0:
            if flux_idx == 0:
                np.save(path_ps / Path(f'pix_lon1.npy'), pix_lon)
                np.save(path_ps / Path(f'pix_lat1.npy'), pix_lat)
            elif flux_idx == 1:
                np.save(path_ps / Path(f'pix_lon2_{distance_factor}.npy'), pix_lon) # degree
                np.save(path_ps / Path(f'pix_lat2_{distance_factor}.npy'), pix_lat) # degree

        delta_q_map[ps_pix_idx] = qflux
        delta_u_map[ps_pix_idx] = uflux

    hp.gnomview(delta_q_map, rot=[lon, lat, 0])
    plt.show()

    m_ps = hp.smoothing(map_in=[delta_t_map, delta_q_map, delta_u_map], fwhm=np.deg2rad(beam)/60, pol=True)
    hp.gnomview(m_ps[1], rot=[0, 0, 0])
    plt.show()

    np.save(f'./data/ps/ps{distance_factor}.npy', m_ps)
    return m_ps

def gen_map(lmax, freq, beam, nside, distance_factor, rlz_idx=0, qflux0=6000, uflux0=4000, qflux1=4000, uflux1=-6000):
    npix = hp.nside2npix(nside=nside)

    # m_ps = gen_ps(freq, beam, nside, distance_factor, rlz_idx=rlz_idx, qflux0=qflux0, uflux0=uflux0, uflux1=uflux1)
    m_ps = np.load('./data/ps/ps3.npy')

    noise_seed = np.load('../seeds_noise_2k.npy')
    fg_seed = np.load('../seeds_fg_2k.npy')
    cmb_seed = np.load('../seeds_cmb_2k.npy')

    nstd = np.load(f'../../FGSim/NSTDNORTH/{nside}/{freq}.npy')
    np.random.seed(seed=noise_seed[rlz_idx])
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3,npix))

    cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    np.random.seed(seed=cmb_seed[rlz_idx])
    # cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=1999)
    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=3*nside - 1)

    cls_fg = gen_fg_cl(freq=freq)
    np.random.seed(seed=fg_seed[rlz_idx])
    fg_iqu = hp.synfast(cls_fg, nside=nside, fwhm=0, new=True, lmax=lmax)

    pcfn = noise + m_ps + cmb_iqu + fg_iqu

    if rlz_idx == 0:
        path_m = Path('./data/map')
        path_m.mkdir(exist_ok=True, parents=True)
        np.save(f'./data/map/pcfn_{distance_factor}_{rlz_idx}.npy', pcfn)

    hp.gnomview(pcfn, rot=[0, 0, 0])
    plt.show()
    return pcfn



if __name__=="__main__":
    # gen_ps(freq=30, beam=67, nside=2048, distance_factor=3, rlz_idx=0)
    gen_map(lmax=500, freq=30, beam=67, nside=2048, distance_factor=3, rlz_idx=0)



