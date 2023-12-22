from scipy.stats import chi2

# Given values
chi_squared_statistic = 4400
alpha = 0.05

# Calculating the p-value from the chi-squared statistic
p_value = chi2.sf(chi_squared_statistic, df=4207)  # sf (survival function) is 1 - cdf
print(f'{p_value=}')

