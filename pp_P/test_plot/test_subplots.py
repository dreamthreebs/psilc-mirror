import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

# 生成一些示例数据
nside = 32
npix = hp.nside2npix(nside)
data = np.random.normal(loc=0, scale=1, size=(npix,))
# data = hp.ud_grade(hp.read_map('example_map.fits'), nside)


hp.mollview(data, sub=221)
plt.title('Map Title', pad=0)
hp.graticule()
hp.mollview(data, sub=222)
hp.mollview(data, sub=223)
hp.mollview(data, sub=224)

plt.show()
