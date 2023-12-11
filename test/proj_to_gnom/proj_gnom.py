import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

def make_a_ps(lon, lat, nside, beam, lmax, norm=1e6):
    '''longitude and latitude in degree'''
    npix = hp.nside2npix(nside)
    m = np.zeros(npix)
    
    ipix = hp.ang2pix(nside=nside, theta=lon, phi=lat, lonlat=True)
    vec = hp.pix2vec(nside=nside, ipix=ipix)
    print(f'{ipix=}')
    m[ipix] = norm
    
    sm_m = hp.smoothing(m, fwhm=np.deg2rad(beam)/60, lmax=lmax)
    
    return sm_m

def my_vec2pix_func(x, y, z):
    ipix = hp.vec2pix(nside=nside, x=x, y=y, z=z)
    return ipix


if __name__=="__main__":

    nside = 512
    lmax = 2000
    npix = hp.nside2npix(nside)
    beam = 63
    sigma = np.deg2rad(beam) / 60 / np.sqrt(8*np.log(2))
    
    # nstd = np.load('../../FGSim/NSTDNORTH/40.npy')[0]
    # noise = nstd * np.random.normal(0,1,(npix))

    m = make_a_ps(lon=13, lat=59, nside=nside, beam=beam, lmax=lmax)

    hp.gnomview(m, rot=[13, 59,0])
    plt.show()

    # generate gnomproject object
    gproj_obj = hp.projector.GnomonicProj()

    # projection map information
    print('get_center output:', gproj_obj.get_center(lonlat=True))
    print('get_extent output:', gproj_obj.get_extent())
    print('get_fov output:', gproj_obj.get_fov())
    print('get_proj_plane_info:', gproj_obj.get_proj_plane_info())
    print('array_info:', gproj_obj.arrayinfo)

    
    # get projection map especially its value
    proj_m = gproj_obj.projmap(m, my_vec2pix_func)
    print(f'{proj_m=}')
    print(f'{proj_m.shape=}')

    # get position in the projection plane
    pos_xy = gproj_obj.ij2xy() # the output is a tuple
    print(f'{pos_xy=}')
    print(f'{pos_xy[0].shape=}')

    idx = gproj_obj.xy2ij(pos_xy)
    print(f'{idx=}')

    plt.imshow(proj_m, extent=gproj_obj.get_extent(), origin='lower')
    plt.show()

    plt.contourf(id)


