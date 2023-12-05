import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre

# Create an array of x values
x = np.linspace(-1, 1, 400)

# Plot Legendre polynomials of different orders
for n in range(6):  # Plotting first four Legendre polynomials
    Pn = legendre(n)
    plt.plot(x, Pn(x), label=f'$P_{n}(x)$')

plt.legend()
plt.xlabel('$x$')
plt.ylabel('$P_n(x)$')
plt.title('Legendre Polynomials')
plt.show()

