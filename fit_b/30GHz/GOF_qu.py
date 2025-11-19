import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import glob
from scipy.stats import norm, chi2
from fit_qu_no_const import FitPolPS

from config import freq, nside

factor = hp.nside2pixarea(nside=nside)
# factor = 1
df = pd.read_csv(f'./mask/{freq}.csv')

factor_convert = FitPolPS.mJy_to_uKCMB(1/factor, freq) # 1mJy = how many uK(arbitrary)

for flux_idx in range(0,10):
    print(f'{flux_idx=}')
    flux_p = df.at[flux_idx, 'pflux']
    flux_q = df.at[flux_idx, 'qflux']
    flux_u = df.at[flux_idx, 'uflux']

    # Assuming data is loaded or generated here
    # data = np.random.normal(loc=0, scale=1, size=10000)  # Example data
    # data = np.load('./PSCMBNOISE/normalize_noise_1000/idx_1/norm_beam.npy')
    P_list = []
    Q_list = []
    U_list = []
    phi_list = []
    
    # pos_list = glob.glob(f'./params/0/fit_1/phi_*.npy')
    
    # pos_list = glob.glob(f'./fit_res/pcn_params/idx_0/fit_P_99.npy')
    
    # for p in pos_list:
    #     P = np.load(p)
    #     P_list.append(P)
    
    # print(f'{P_list=}')
    
    for rlz_idx in range(0,200):
        P = np.load(f'./parameter/th_have_all/fit_P_{rlz_idx}.npy')
        phi = np.load(f'./parameter/th_have_all/fit_phi_{rlz_idx}.npy')
        # print(f"{P=}")
        # phi = np.load(f'./params/0/fit_2/phi_{rlz_idx}.npy')
        Q = P * np.cos(phi) / factor_convert
        U = P * np.sin(phi) / factor_convert
        P_list.append(P)
        Q_list.append(Q)
        U_list.append(U)
        # phi_list.append(phi)

    data = np.asarray(Q_list)
    print(f'{data.shape=}')

    # data_mean = np.mean(data, axis=0)
    # data_std = np.std(data, axis=0)
    # print(f'{data_mean=}')
    # print(f'{data_std=}')
    # SEM = data_std / np.sqrt(1000)
    # t = (data_mean - 757.28) / SEM
    # print(f'{t=}')
    
    # Define the number of bins
    bin_count = 10
    
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
    plt.plot(x, scaled_pdf, 'r--', linewidth=2, label=f'Fit (mu={mu:.2f}, std={std:.2f})')
    
    # mu_ref = np.sqrt(250**2 + 500**2)
    mu_ref = flux_q # 757.28
    std_ref =  23 # 23.782
    
    scaled_ref_pdf = norm.pdf(x, mu_ref, std_ref) * len(data) * np.diff(bin_edges)[0]
    # plt.plot(x, scaled_ref_pdf, 'k', linewidth=2, label=f'Ref (mu={mu_ref:.2f}, std={std_ref:.2f})')
    plt.axvline(x=mu_ref, color='purple', linewidth=2, label=f'Input value: {mu_ref:.2f}')
    
    plt.title(f"Fit results: mu = {mu:.2f}mJy, std = {std:.2f}mJy\nChi-squared test: χ² = {chi_squared_stat:.2f}, p-value = {p_value:.3f}")
    plt.xlabel('Point source amplitude [mJy]')
    plt.ylabel("Counts")
    # plt.legend()
    plt.savefig(f'/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20251028/th_{freq}.png', dpi=300)
    plt.show()
    
    # Print the chi-squared test result
    print(f"Chi-squared test: χ² = {chi_squared_stat:.2f}, p-value = {p_value:.3f}")









