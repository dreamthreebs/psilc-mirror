from astropy import units as u
from astropy.cosmology import Planck15

freq = 40 * u.GHz
equiv = u.thermodynamic_temperature(freq, Planck15.Tcmb0)
print((1. * u.mK).to(u.mJy / u.sr, equivalencies=equiv))
