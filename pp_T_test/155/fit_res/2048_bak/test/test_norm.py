# import numpy as np
# import healpy as hp
# from scipy.stats import norm
# import matplotlib.pyplot as plt

# norm_beam = np.load('../PSCMBNOISE/1.5/idx_0/norm_beam.npy')
# print(f'{norm_beam=}')

# plt.hist(norm_beam, bins=20, density=True)
# # plt.show()

# mu, std = norm.fit(norm_beam)
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, mu, std)
# plt.plot(x, p, 'k', linewidth=2)
# title = "Fit results: mu = %.7f,  std = %.7f" % (mu, std)
# plt.title(title)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate synthetic data
data = np.random.normal(loc=0, scale=1, size=1000)

# Define the number of bins
bin_count = 30

# Calculate the histogram without density normalization
hist_counts, bin_edges = np.histogram(data, bins=bin_count)

# Calculate bin centers
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Fit the data to a normal distribution to get mean and standard deviation
mu, std = norm.fit(data)

# Calculate expected frequencies for each bin using the PDF scaled by the total count and bin width
expected_freq = norm.pdf(bin_centers, mu, std) * sum(hist_counts) * np.diff(bin_edges)

# Plot the histogram and the PDF
plt.hist(data, bins=bin_count, density=True, alpha=0.6, color='g')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.title(f"Fit results: mu = {mu:.2f}, std = {std:.2f}")
plt.show()

# Perform the Chi-Squared test
from scipy.stats import chisquare
chi_stat, p_value = chisquare(f_obs=hist_counts, f_exp=expected_freq)

print(f"Chi-Squared Statistic: {chi_stat}, P-value: {p_value}")

