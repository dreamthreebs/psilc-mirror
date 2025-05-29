import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt

def check_mask_second_der(mask):
    wlm = hp.map2alm(mask)
    nside = hp.get_nside(mask)
    ls = np.arange(3*nside)
    wlm1 = hp.almxfl(wlm, np.sqrt(ls*(ls+1)))
    wlm2 = hp.almxfl(wlm, np.sqrt((ls-1)*ls*(ls+1)*(ls+2)))
    mask1 = hp.alm2map_spin([wlm1, 0*wlm1], nside, 1, 3*nside-1)
    mask2 = hp.alm2map_spin([wlm2, 0*wlm2], nside, 2, 3*nside-1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.axes(axes[0][0])
    hp.orthview(mask1[0], title=r'$\partial_\theta w$', hold=True, rot=[100,50,0])
    plt.axes(axes[0][1])
    hp.orthview(mask1[1], title=r'$\partial_\varphi w$', hold=True, rot=[100,50,0])
    plt.axes(axes[1][0])
    hp.orthview(mask2[0], title=r'$\partial^2_Q w$', hold=True, rot=[100,50,0])
    plt.axes(axes[1][1])
    hp.orthview(mask2[1], title=r'$\partial^2_U w$', hold=True, rot=[100,50,0])
    plt.show()

def calc_fsky(mask):
    """
    Return the effective sky fraction f_sky = mean(mask**2).

    Works for binary or apodized HEALPix masks.
    """
    return np.sum(mask**2) / np.size(mask)

def gen_circular_mask(nside):
    vec = hp.ang2vec(theta=100, phi=50,lonlat=True)
    pix_disc = hp.query_disc(nside=nside, vec=vec, radius=np.deg2rad(23))
    mask = np.zeros(hp.nside2npix(nside))
    mask[pix_disc] = 1
    return mask



if __name__ == "__main__":
    # apo_mask = np.load('../../../psfit/fitv4/fit_res/2048/ps_mask/new_mask/apo_C1_3_apo_3_apo_3.npy')
    bin_mask = gen_circular_mask(nside=2048)
    apo_mask = nmt.mask_apodization(mask_in=bin_mask, aposize=5)
    np.save('./apo_mask_test.npy', apo_mask)
    
    # apo_mask = np.load(f"../inpainting/new_mask/apo_ps_mask.npy")
    # check_mask_second_der(mask=apo_mask)
    fsky = calc_fsky(mask=apo_mask)
    print(f"{fsky=}")
    hp.orthview(apo_mask, rot=[100,50,0])
    plt.show()
