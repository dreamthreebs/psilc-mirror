import numpy as np
import matplotlib.pyplot as plt

method_list = ['GLSPF', 'Masking', 'Inpainting', 'Simulation with PS', 'Simulation without PS']

# Define mean r values (corresponding to reversed methods)
r_mean_list = [2.25e-3, 3.08e-3, 1.06e-2, 3.58e-3, 1.71e-3]
# Define asymmetric errors (lower and upper limits) in reversed order
r_lower_list = [-2.57e-3, -2.01e-3, 4.41e-3, -1.27e-3, -3.01e-3]
r_upper_list = [7.26e-3, 8.14e-3, 1.62e-2, 8.47e-3, 6.62e-3]

# Compute the lower and upper error bars
r_err_lower = [r_mean - r_low for r_mean, r_low in zip(r_mean_list, r_lower_list)]
r_err_upper = [r_up - r_mean for r_mean, r_up in zip(r_mean_list, r_upper_list)]

# Convert to error tuple format
r_errs = [r_err_lower, r_err_upper]

# Define colors for each method
colors = ['green', 'orange', 'red', 'blue', 'purple']  # Colors still match each method

# Create the figure and axis
fig, ax = plt.subplots(figsize=(7, 4))

# Adjust y-axis positions manually to reduce gap
y_positions = np.linspace(0, len(method_list) - 1, len(method_list))  # Evenly spaced but closer together

# Plot each method with a different color
for i, (r_mean, r_err_lower, r_err_upper, color) in enumerate(zip(r_mean_list, r_err_lower, r_err_upper, colors)):
    ax.errorbar(r_mean, y_positions[i], xerr=[[r_err_lower], [r_err_upper]], fmt='o', capsize=5, color=color, label=method_list[i])

# Set y-axis ticks and labels
ax.set_yticks(range(len(method_list)))
ax.set_yticklabels(method_list, fontsize=10)

# Set axis labels
ax.set_xlabel('Tensor to scalar ratio r')

# Use log scale for x-axis
ax.set_xscale('log')
ax.set_xlim(1e-3, 2e-2)

# Show grid
ax.grid(True, which="both", linestyle="--", alpha=0.6)

# Show legend
# ax.legend(loc='lower right')
plt.tight_layout()

plt.savefig("/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20250609/r_constraints.png", dpi=300, bbox_inches="tight")

plt.show()

