import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2

# Assuming data is loaded or generated here
# data = np.random.normal(loc=0, scale=1, size=10000)  # Example data
# data = np.load('./PSCMBNOISE/normalize_noise_1000/idx_1/norm_beam.npy')
P_list = []
phi_list = []
for rlz_idx in range(1,4599):
    P = np.load(f'./params/P_{rlz_idx}.npy')
    phi = np.load(f'./params/phi_{rlz_idx}.npy')
    P_list.append(P)
    phi_list.append(phi)

data = np.asarray(phi_list)
print(f'{data=}')
data_mean = np.mean(data, axis=0)
data_std = np.std(data, axis=0)
print(f'{data_mean=}')
print(f'{data_std=}')
SEM = data_std / np.sqrt(1000)
t = (data_mean - np.arctan2(727.84, -209.09)) / SEM
print(f'{t=}')



# Define the number of bins
bin_count = 15

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
plt.plot(x, scaled_pdf, 'r--', linewidth=2, label=f'Fit (mu={mu:.4f}, std={std:.4f})')

mu_ref = np.arctan2(727.84, -209.09)
std_ref = 0.0323

scaled_ref_pdf = norm.pdf(x, mu_ref, std_ref) * len(data) * np.diff(bin_edges)[0]
plt.plot(x, scaled_ref_pdf, 'k', linewidth=2, label=f'Ref (mu={mu_ref:.4f}, std={std_ref:.4f})')

plt.title(f"Fit results: mu = {mu:.4f}, std = {std:.4f}\nChi-squared test: χ² = {chi_squared_stat:.2f}, p-value = {p_value:.3f}")
plt.xlabel('Polarization angle')
plt.ylabel("Counts")
plt.legend()
plt.show()

# Print the chi-squared test result
print(f"Chi-squared test: χ² = {chi_squared_stat:.2f}, p-value = {p_value:.3f}")




