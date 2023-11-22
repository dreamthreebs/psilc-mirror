import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd

df30 = pd.read_csv('./sort_by_iflux/30.csv')

lon = df30.at[0, 'lon']
lat = df30.at[0, 'lat']





