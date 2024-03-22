import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2

# Assuming data is loaded or generated here
# data = np.random.normal(loc=0, scale=1, size=10000)  # Example data
data = np.load('./PSCMBNOISE/1.5/idx_79/norm_beam.npy')

# Define the number of bins
bin_count = 25

# Calculate the histogram as counts
hist_counts, bin_edges = np.histogram(data, bins=bin_count)

# Calculate bin centers
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Fit the data to a normal distribution to get mean and standard deviation
mu, std = norm.fit(data)

# Calculate expected frequencies for each bin using the PDF scaled by the total count and bin width
expected_freq = norm.pdf(bin_centers, mu, std) * len(data) * np.diff(bin_edges)

# Perform the chi-squared test
chi_squared_stat = ((hist_counts - expected_freq) ** 2 / expected_freq).sum()
p_value = chi2.sf(chi_squared_stat, df=bin_count-1-2)  # df = number of bins - 1 - number of estimated parameters

# Plot the histogram as counts and the expected PDF scaled to the histogram
plt.bar(bin_centers, hist_counts, width=bin_edges[1] - bin_edges[0], color='g', alpha=0.6)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
scaled_pdf = norm.pdf(x, mu, std) * len(data) * np.diff(bin_edges)[0]
plt.plot(x, scaled_pdf, 'r--', linewidth=2, label=f'Fit (mu={mu:7f}, std={std:7f})')

mu_ref = 0.004156
std_ref = 0.000183
scaled_ref_pdf = norm.pdf(x, mu_ref, std_ref) * len(data) * np.diff(bin_edges)[0]
plt.plot(x, scaled_ref_pdf, 'k', linewidth=2, label=f'Ref (mu={mu_ref}, std={std_ref})')

plt.title(f"Fit results: mu = {mu:.7f}, std = {std:.7f}\nChi-squared test: χ² = {chi_squared_stat:.2f}, p-value = {p_value:.3f}, flux_idx=2")
plt.xlabel('Point source amplitude')
plt.ylabel("Counts")
plt.legend()
plt.show()

# Print the chi-squared test result
print(f"Chi-squared test: χ² = {chi_squared_stat:.2f}, p-value = {p_value:.3f}")

