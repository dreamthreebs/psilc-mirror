import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

freq = 215
lmax = 350
fold = 1.8
l = np.arange(lmax+1)
m_in = hp.read_map(f'../inputcmb/{fold}/{freq}.fits', field=0)
cl_in = hp.anafast(m_in, lmax=lmax)
plt.semilogy(l*(l+1)*cl_in/(2 * np.pi), label='cmb input ')

for fold in [0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]:
    m_out = hp.read_map(f'./{fold}/{freq}.fits', field=0)
    cl_out = hp.anafast(m_out, lmax=lmax)

    # hp.mollview(m_out)
    # plt.show()

   
    m_diff = m_in - m_out
    cl_diff = hp.anafast(m_diff, lmax=lmax)
    
    # plt.semilogy(l*(l+1)*cl_out/(2 * np.pi), label=f"cmb output {fold=}")
    plt.semilogy(l*(l+1)*cl_diff/(2 * np.pi), label=f"map difference {fold=}")
    
    # plt.semilogy(l*(l+1)*np.abs(cl_in-cl_out)/(2 * np.pi), label="dl difference")
    
plt.legend()
plt.show()

