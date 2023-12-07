import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from numpy.polynomial.legendre import Legendre
import pickle

def count_local_extrema(l, x_range, num_points=10000):
    x = np.linspace(x_range[0], x_range[1], num_points)
    leg_poly = Legendre.basis(l)
    leg_poly_der = np.gradient(leg_poly(x), x)
    peaks, _ = find_peaks(leg_poly(x))
    valleys, _ = find_peaks(-leg_poly(x))
    extrema = np.sort(np.concatenate((peaks, valleys)))
    return len(extrema)

def generate_interp_funcs(lmax, x_range, total_points):
    interp_funcs = {}
    for l in range(lmax+1):
        extrema_count = count_local_extrema(l, x_range)
        print(f'{l=}, {extrema_count=}')
        interp_funcs_per_l = []
        if extrema_count == 0:
            # Use cubic interpolation over the entire range
            x = np.linspace(x_range[0], x_range[1], total_points)
            y = Legendre.basis(l)(x)
            interp_func = interp1d(x, y, kind='cubic')
            interp_funcs_per_l.append((interp_func, x_range))
        else:
            # Calculate weights for each segment
            num_segments = extrema_count + 1
            print(f'{num_segments=}')
            segment_weights = [i / sum(range(1, num_segments + 1)) for i in range(1, num_segments + 1)]

            # Allocate points to each segment based on weights
            segment_points = [int(total_points * weight) for weight in segment_weights]

            # Ensure total sum of points remains constant
            segment_points[-1] += total_points - sum(segment_points)
            print(f'{segment_points=}')

            # Interpolate and plot each segment
            x_segments = np.linspace(x_range[0], x_range[1], num_segments + 1)
            for i in range(extrema_count + 1):
                segment_x_range = (x_segments[i], x_segments[i + 1])
                x_seg = np.linspace(segment_x_range[0], segment_x_range[1], segment_points[i])
                y_seg = Legendre.basis(l)(x_seg)
                interp_func = interp1d(x_seg, y_seg, kind='cubic')
                interp_funcs_per_l.append((interp_func, segment_x_range))
        interp_funcs[l] = interp_funcs_per_l
    return interp_funcs

def evaluate_interp_func(l, x, interp_funcs):
    for interp_func, x_range in interp_funcs[l]:
        if x_range[0] <= x <= x_range[1]:
            return interp_func(x)
    raise ValueError(f"x = {x} is out of the interpolation range for l = {l}")

lmax = 500  # Maximum degree of Legendre polynomial
x_range = (0.998, 1)
total_points = 1000  # Total number of points to be distributed

interp_funcs = generate_interp_funcs(lmax, x_range, total_points)

# Pickle the dictionary of interpolation functions
with open('lgd_itp_funcs500.pkl', 'wb') as f:
    pickle.dump(interp_funcs, f)

# Usage Example: Load and use the interpolation functions

# with open('lgd_itp_funcs1500', 'rb') as f:
#     loaded_interp_funcs = pickle.load(f)
#     # Example: Evaluate the interpolated function for l=3 at x=0.9999
#     l = 100
#     x_value = 0.99998
#     interpolated_value = evaluate_interp_func(l, x_value, loaded_interp_funcs)
#     print(interpolated_value)





