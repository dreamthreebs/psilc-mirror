from scipy.stats import chi2

# Calculate the area within a 2.5 sigma ellipse in a 2D Gaussian distribution
# The chi-squared distribution with 2 degrees of freedom is used, as it's related to the bivariate normal distribution
area_2_5_sigma_2d = chi2.cdf(2.9**2, 2)  # Square of 2.5 for the chi-squared distribution and 2 degrees of freedom
area_2_5_sigma_2d_percentage = area_2_5_sigma_2d * 100
print(area_2_5_sigma_2d_percentage)

