import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lon = np.rad2deg(np.pi)
lat = np.rad2deg(np.pi/2)

def adjust_latitude(lat):
    lat = lat % 360
    if lat < -90:
        lat = -180 - lat
    if (lat > 90) and (lat <= 270):
        lat = 180 - lat
    elif lat > 270:
        lat = lat - 360

    return lat

# Example values
latitudes = [30, -91, 95, 187, 365, -380, -171, -268, -435, -275, 275]

# Adjusting the latitudes and longitudes
adjusted_values = [adjust_latitude(lat) for lat in latitudes]
print(f"{latitudes=}")
print(f"{adjusted_values=}")

print(f'{lon=}, {lat=}')
vec = hp.ang2vec(theta=np.nan, phi=lat, lonlat=True)
print(f'{vec}')

# vec = hp.ang2vec(theta=359, phi=lat, lonlat=True)
# print(f'{vec}')
