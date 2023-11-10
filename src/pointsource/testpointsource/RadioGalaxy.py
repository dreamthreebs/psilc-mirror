import pysm3
import pysm3.units as u
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

sky = pysm3.Sky(nside=256, preset_strings=["rg1"])

map_100GHz = sky.get_emission(100 * u.GHz)

