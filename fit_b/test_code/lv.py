import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator

filename = './Cascade_B_gammas.dat'


def dNdx_exp(x, eps_f, cas_order='n=1'):
    #x_value = ene/mx #energy expressed in terms of x= E/m_DM
    log10x = np.log10(x)
    spec_data = pd.read_table('./Cascade_B_gammas.dat', sep="\s+", usecols=['EpsF', 'Log[10,x]', cas_order])
    # print(f'{spec_data=}')
    # epfs_fs = np.unique(spec_data['EpsF'])
    epfs_fs = np.unique(spec_data['EpsF'])[::-1]
    print(f'{epfs_fs[2]=}')
    log10xs = np.unique(spec_data['Log[10,x]'])
    print(f'{log10xs[3]=}')
    # SMspectrum = np.array(spec_data[cas_order]).reshape(len(epfs_fs), len(log10xs), order='C')
    # print(f'{SMspectrum.T[2,3]=}')
    SMspectrum = np.array(spec_data[cas_order]).reshape(len(log10xs), len(epfs_fs), order='C')
    print(f'{SMspectrum.T[2,3]=}')
    # print(f'{SMspectrum=}')
    interpolated_spectrum = RegularGridInterpolator((log10xs, epfs_fs), SMspectrum.T, method='linear', bounds_error=False, fill_value=None)#, fill_value=0)
    eps = interpolated_spectrum((log10x, eps_f))/(np.log(10)*x)
    # print(f'{eps=}')

    return eps

# dNdx_exp(x=3, eps_f=10)
with open(filename) as f:
    lines = (line for line in f if not line.startswith('#'))
    data = np.genfromtxt (lines, names = True ,dtype = None)

epsvals = data["EpsF"]
eps_f = 0.5 #should be 2 * 1.77 /10 = 0.354
index = np.where(np.abs( (epsvals - eps_f) / eps_f) < 1.e-3)
xvals = 10**(data["Log10x"][index])

flux = [data["n"+str(i)][index]/(np.log(10)*xvals) for i in range(1,7)]
loadspec = [interp1d(xvals,flux[i]) for i in range(6)]
def dNdx(x,step):
    fluxval = loadspec[step-1](x)
    if (x>1 or fluxval<0):
        return 0
    else:
        return fluxval

flux_inp = dNdx_exp(xvals, eps_f=0.5)
flux_inp2 = dNdx_exp(xvals, eps_f=0.01)
plt.plot(xvals,xvals**2*flux_inp,label='flux 0.5',color='red')
# plt.plot(xvals,xvals**2*flux_inp2,label='flux 0.5',color='black')
plt.plot(xvals,[x**2*dNdx(x,1) for x in xvals],label='n=1',color='Purple', linestyle='--')
#plt.plot(xvals,[x**2*dNdx(x,2) for x in xvals],label='n=2',color='Blue')
#plt.plot(xvals,[x**2*dNdx(x,3) for x in xvals],label='n=3',color='Green')
#plt.plot(xvals,[x**2*dNdx(x,4) for x in xvals],label='n=4',color='Pink')
#plt.plot(xvals,[x**2*dNdx(x,5) for x in xvals],label='n=5',color='Orange')
#plt.plot(xvals,[x**2*dNdx(x,6) for x in xvals],label='n=6',color='Red')
plt.title('Ex. 1: Photon Cascade Spectra into W Bosons for $\epsilon_f=0.1$',fontsize=14)
plt.xscale('log')
plt.xlabel('$x$', fontsize=18)
plt.ylabel('$x^2 dN / dx$', fontsize=18)
plt.ylim([0.0,0.12])
plt.xlim([10**-6,1])
plt.legend(fontsize=12,loc=2)
plt.show()
