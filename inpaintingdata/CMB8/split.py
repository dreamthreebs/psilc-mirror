import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = hp.read_map('./40.fits', field=1)
hp.write_map('./Q40.fits', m, overwrite=True)

