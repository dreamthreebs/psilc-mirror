import numpy as np
import healpy as hp

from fit_qu import FitPolPS

nside = 2048
factor = hp.nside2pixarea(nside=nside)

print(FitPolPS.mJy_to_uKCMB(1, 30)/factor)
