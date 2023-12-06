import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from numpy.polynomial.legendre import Legendre

def find_local_extrema(l, x_range, num_points=10000):
    # Generate points within the range
    x = np.linspace(x_range[0], x_range[1], num_points)
    
    # Compute Legendre polynomial and its derivative
    leg_poly = Legendre.basis(l)
    leg_poly_der = np.gradient(leg_poly(x), x)

    # Find peaks and valleys (maxima and minima)
    peaks, _ = find_peaks(leg_poly(x))
    valleys, _ = find_peaks(-leg_poly(x))

    # Combine and sort extrema
    extrema_indices = np.sort(np.concatenate((peaks, valleys)))
    extrema_x = x[extrema_indices]

    # Filter extrema within the range
    return extrema_x[(extrema_x > x_range[0]) & (extrema_x < x_range[1])]

def interpolate_legendre(l, x_range, extrema_x, total_points):
    # Create segments based on extrema
    segments = [(x_range[0], extrema_x[0])] + \
               [(extrema_x[i], extrema_x[i + 1]) for i in range(len(extrema_x) - 1)] + \
               [(extrema_x[-1], x_range[1])]

    # Calculate the total length of each segment
    segment_lengths = [seg[1] - seg[0] for seg in segments]
    total_length = sum(segment_lengths)

    # Allocate points based on segment length
    points_per_segment = [max(round(total_points * length / total_length), 1) for length in segment_lengths]
    points_per_segment[0] = points_per_segment[1]  # Adjust first segment to match the second

    for i in range(len(points_per_segment)):
        if points_per_segment[i] < 4:
            points_per_segment[i] = 4


    # Normalize the total number of points
    while sum(points_per_segment) != total_points:
        for i in range(len(points_per_segment)):
            if sum(points_per_segment) < total_points:
                points_per_segment[i] += 1
            elif sum(points_per_segment) > total_points:
                points_per_segment[i] = max(points_per_segment[i] - 1, 1)

    for i, seg in enumerate(segments):
        num_points = points_per_segment[i]
        x_seg = np.linspace(seg[0], seg[1], num_points)
        y_seg = Legendre.basis(l)(x_seg)

        # Choose interpolation method based on the number of points
        interp_method = 'cubic' if num_points >= 4 else 'linear'
        interp_func = interp1d(x_seg, y_seg, kind=interp_method)

        plt.plot(x_seg, interp_func(x_seg), label=f'Segment {i+1}')

    plt.title(f'Interpolation of Legendre Polynomial of Degree {l}')
    plt.legend()
    plt.show()

# Example usage
l = 1000  # Degree of Legendre polynomial
x_range = (0.9998, 1)
total_points = 1000  # Total number of points to be distributed

# Find local extrema in the range
extrema_x = find_local_extrema(l, x_range)

# Interpolate Legendre polynomial with points allocated based on segment length
interpolate_legendre(l, x_range, extrema_x, total_points)

