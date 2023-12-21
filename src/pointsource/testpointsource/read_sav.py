import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.io import readsav

data = readsav('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/40GHz/strongradiops_cat_40GHz.sav',python_dict=True, verbose=True)

