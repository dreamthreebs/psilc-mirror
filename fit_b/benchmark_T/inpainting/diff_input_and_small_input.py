import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m_in = hp.read_map('./input/0.fits')
m_out = hp.read_map('./output/0.fits')
m_in_edge = hp.read_map('./input_small_sky/0.fits')
m_out_edge = hp.read_map('./output_small_sky/0.fits')

hp.orthview(m_in, rot=[100,50,0], half_sky=True)
hp.orthview(m_in_edge, rot=[100,50,0], half_sky=True)
hp.orthview(m_out_edge, rot=[100,50,0], half_sky=True, title='out edge')
hp.orthview(m_in - m_in_edge, rot=[100,50,0], half_sky=True, title='residual')
hp.orthview(m_in_edge - m_out_edge, rot=[100,50,0], half_sky=True, title='in - out')
hp.orthview(m_out - m_out_edge, rot=[100,50,0], half_sky=True, title='out - out edge')


plt.show()

