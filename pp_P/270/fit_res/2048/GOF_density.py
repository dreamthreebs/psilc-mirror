import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2

# Load your data
data = np.load('./PSCMBNOISE/1.5/idx_2/u_amp.npy')[1:100]
print(f'{data.shape=}')

# Define the number of bins
bin_count = 15

# Calculate the histogram without density normalization
hist_counts, bin_edges = np.histogram(data, bins=bin_count, density=False)

# Calculate bin centers
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Fit the data to a normal distribution to get mean and standard deviation
mu, std = norm.fit(data)

# Calculate expected frequencies for each bin using the PDF scaled by the total count and bin width
expected_freq = norm.pdf(bin_centers, mu, std) * sum(hist_counts) * np.diff(bin_edges)

# Ensure that expected frequencies are sufficient for the chi-squared test
if np.any(expected_freq < 5):
    print("Warning: Some expected frequencies are less than 5, which may affect the chi-squared test's validity.")

# Perform the chi-squared test
chi_squared_stat = ((hist_counts - expected_freq) ** 2 / expected_freq).sum()
p_value = chi2.sf(chi_squared_stat, df=bin_count-1-2) # df = number of bins - 1 - number of estimated parameters

# Plot the histogram and the PDF
plt.hist(data, bins=bin_count, density=True, alpha=0.6, color='g')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.xlabel('point source amplitude')
plt.ylabel('pdf')
plt.title(f"Fit results: mu = {mu:.7f}, std = {std:.7f}\nChi-squared test: χ² = {chi_squared_stat:.2f}, p-value = {p_value:.3f}, bin_count = {bin_count}")
plt.show()

# Print the chi-squared test result
print(f"Chi-squared test: χ² = {chi_squared_stat:.2f}, p-value = {p_value:.3f}, bin_count= {bin_count}")


