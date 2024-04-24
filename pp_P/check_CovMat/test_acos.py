import numpy as np

# Generate linearly spaced values from -1 to 1
input_values = np.linspace(-1, -0.998, num=10000)

# Calculate the arc cosine (theta) of these values
theta = np.arccos(input_values)

# Recalculate cosine of the angles obtained from acos
cos_theta = np.cos(theta)

# Calculate the difference between the original and recalculated values
differences = np.abs(cos_theta - input_values)

# Print max and mean differences to evaluate precision
print(f"Maximum difference: {np.max(differences)}")
print(f"Mean difference: {np.mean(differences)}")

# Optionally, plot the differences to visualize precision across the range
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(input_values, differences, label='Difference')
plt.xlabel('Input values (cos θ)')
plt.ylabel('Difference')
plt.title('Precision of np.arccos across its domain')
plt.legend()
plt.grid(True)
plt.show()

