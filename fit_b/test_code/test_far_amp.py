import numpy as np
import matplotlib.pyplot as plt

sigma = 1
r = np.linspace(start=0, stop=10)
print(f'{r=}')

print(f'{np.sqrt(8*np.log(2))=}')


tau = 1 / (r**2) * (np.exp(-r**2/2) * (1 + r**2/2) - 1)
rho = 1 / (2 * np.pi) * np.exp(-r**2/2)

plt.plot(r, tau)
plt.plot(r, rho)
plt.scatter(r[24], tau[24], label='2*fwhm')
plt.legend()
plt.xlabel('r [sigma]')
# plt.ylabel('$\\tau(r)$')
plt.ylabel('$\\tau(r) or \\rho(r)$')
plt.show()

