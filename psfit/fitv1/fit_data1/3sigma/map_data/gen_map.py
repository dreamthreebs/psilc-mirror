import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('../../../../../FGSim/STRPSCMBFGNOISE/40.npy')[0]
hp.write_map('./maps.fits', m)
