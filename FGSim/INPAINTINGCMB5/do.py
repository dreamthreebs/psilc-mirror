import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from pathlib import Path


def Tmap_test(freq):
    lmax = 350
    nside = 512
    m = np.load(f'../CMB5/{freq}.npy')
    mask = hp.read_map(f'../../FG5/strongps/psmaskfits/{fold}/{freq}.fits')
    # B = hp.alm2map(hp.map2alm(m, lmax=lmax)[2], nside=nside)
    
    """
    try on T map
    """
    
    T = m[0]
    # print(f'{m.shape=}')
    hp.write_map(f'./inputcmb/{fold}/{freq}.fits', T, overwrite=True)
    # hp.mollview(T * mask, title=f'{fold=}, {freq=}')
    # plt.show()

if __name__=='__main__':
    fold = 0.9
    directory = Path(f'./inputcmb/{fold}')
    directory.mkdir(parents=True, exist_ok=True)

    for freq in [40, 95, 155, 215, 270]:
        print(f'{freq=}')
        Tmap_test(freq)


