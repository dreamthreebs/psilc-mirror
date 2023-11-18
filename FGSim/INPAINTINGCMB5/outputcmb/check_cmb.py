import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

freq = 215
lmax = 350
# fold = 1.8
l = np.arange(lmax+1)
m_in = hp.read_map(f'../inputcmb/smoothcmb/{freq}.fits', field=0)
cl_in = hp.anafast(m_in, lmax=lmax)
plt.semilogy(l*(l+1)*cl_in/(2 * np.pi), label='cmb+ps input ')

m_theory = np.load(f'../../CMB5/215.npy')[0]
cl_theory = hp.anafast(m_theory, lmax=lmax)
plt.semilogy(l*(l+1)*cl_theory/(2 * np.pi), label='cmb theory')

fold = "smoothcmb"
m_out = hp.read_map(f'./smoothcmb/{freq}.fits', field=0)
cl_out = hp.anafast(m_out, lmax=lmax)

# hp.mollview(m_out)
# plt.show()


m_diff = m_theory - m_out
cl_diff = hp.anafast(m_diff, lmax=lmax)

plt.semilogy(l*(l+1)*cl_out/(2 * np.pi), label=f"cmb output {fold=}")
plt.semilogy(l*(l+1)*cl_diff/(2 * np.pi), label=f"map difference {fold=}")

# plt.semilogy(l*(l+1)*np.abs(cl_in-cl_out)/(2 * np.pi), label="dl difference")

plt.legend()
plt.show()

