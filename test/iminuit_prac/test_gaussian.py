import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

# Set the desired number of samples
num_samples = 10
np.random.seed(100)

x_arr = np.linspace(-1,1,100)
print(f'{x_arr=}')

def func(x_arr):
    return 1/(np.sqrt(2*np.pi)) * np.exp(-(x_arr)**2/(2 * 1**2))

y_arr = func(x_arr)
print(f'{y_arr=}')

y_arr_err = np.abs(y_arr) / 100

# plt.show()


y_data_list = []
for i in range(len(x_arr)):
    y_data = np.random.normal(loc=y_arr[i], scale=1/100*y_arr[i])
    print(f'{y_data=}')
    y_data_list.append(y_data)
    
y_data_arr = np.array(y_data_list)

# plt.plot(x_arr, y_arr, label='origin function')
plt.plot(x_arr, y_data_arr, label='samples')
plt.errorbar(x_arr, y_arr, yerr=y_arr_err, capsize=3, label='sigma')
plt.legend()
plt.show()


def gauss_func(x, mu, sigma):
    return 1/(np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x_arr-mu)**2/(2 * sigma**2))

lsq = LeastSquares(x=x_arr, y=y_data_arr, yerror=1/100*y_data_arr, model=gauss_func)

m = Minuit(lsq, mu=0.1, sigma=0.9)

print(m.migrad())
print(m.hesse())

print(f'{m.values}')
print(f'{m.errors}')







