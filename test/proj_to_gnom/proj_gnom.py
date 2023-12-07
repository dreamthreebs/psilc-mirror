import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create a Sample HEALPix Map
# For simplicity, let's create a map with a constant value
nside = 512
npix = hp.nside2npix(nside)
map = np.full(npix, 0.5)  # A map with constant values

# Step 2: Set Up the Gnomonic Projector
# Define the center of the projection and the size of the view
rot = hp.Rotator(coord=['G', 'C'])  # Galactic to Celestial
theta_c, phi_c = np.radians(90), np.radians(0)  # Center at North Galactic Pole
theta_c, phi_c = rot(theta_c, phi_c)  # Convert to Celestial coordinates

# Initialize the Gnomonic projector
gnom_proj = hp.projector.GnomonicProj(rot=[phi_c, theta_c, 0], xsize=200, ysize=200, reso=10)

# Step 3: Project the Map
# Generate the projected image
proj_img = gnom_proj.projmap(map, vec2pix_func=hp.vec2pix)

# Step 4: Visualize the Projection
plt.imshow(proj_img, origin='lower', extent=gnom_proj.get_extent())
plt.xlabel('X [degrees]')
plt.ylabel('Y [degrees]')
plt.title('Gnomonic Projection')
plt.colorbar(label='Map Value')
plt.show()

