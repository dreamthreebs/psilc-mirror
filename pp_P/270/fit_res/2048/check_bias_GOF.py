import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2

# Assuming data is loaded or generated here
# data = np.random.normal(loc=0, scale=1, size=10000)  # Example data
# data = np.load('./PSCMBNOISE/normalize_noise_1000/idx_1/norm_beam.npy')


data = np.load(f'./PSCMBNOISE/check_bias/idx_1/q_amp_1.npy')
for i in range(2,11):
    data1 = np.load(f'./PSCMBNOISE/check_bias/idx_1/q_amp_{i}.npy')
    data = np.concatenate([data, data1])

print(f'{data.shape=}')

# Define the number of bins
bin_count = 30

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
plt.plot(x, scaled_pdf, 'r--', linewidth=2, label=f'Fit (mu={mu:.4e}, std={std:.4e})')

mu_ref = -5.56368e-5
std_ref = 5.42629e-6
scaled_ref_pdf = norm.pdf(x, mu_ref, std_ref) * len(data) * np.diff(bin_edges)[0]
plt.plot(x, scaled_ref_pdf, 'k', linewidth=2, label=f'Ref (mu={mu_ref:.4e}, std={std_ref:.4e})')

plt.title(f"Fit results: mu = {mu:.4e}, std = {std:.4e}\nChi-squared test: χ² = {chi_squared_stat:.2f}, p-value = {p_value:.3f}, flux_idx=1")
plt.xlabel('Point source amplitude')
plt.ylabel("Counts")
plt.legend()
plt.show()

# Print the chi-squared test result
print(f"Chi-squared test: χ² = {chi_squared_stat:.2f}, p-value = {p_value:.3f}")


