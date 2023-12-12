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

    nside = 2048
    lmax = 2000
    npix = hp.nside2npix(nside)
    beam = 63
    sigma = np.deg2rad(beam) / 60 / np.sqrt(8*np.log(2))
    
    # nstd = np.load('../../FGSim/NSTDNORTH/40.npy')[0]
    # noise = nstd * np.random.normal(0,1,(npix))
    ps_lon = 13
    ps_lat = 59

    m = make_a_ps(lon=ps_lon, lat=ps_lat, nside=nside, beam=beam, lmax=lmax)

    # np.save(f'./data/lon{ps_lon}lat{ps_lat}.npy', m)
    hp.gnomview(m, rot=[ps_lon,ps_lat,0])
    plt.show()

    # generate gnomproject object
    gproj_obj = hp.projector.GnomonicProj(rot=[ps_lon, ps_lat,0], xsize=100, ysize=100, reso=1.0)
    # gproj_obj.set_proj_plane_info(xsize=100, ysize=100, reso=1.0)

    # projection map information
    print('get_center output:', gproj_obj.get_center(lonlat=True))
    print('get_extent output:', gproj_obj.get_extent())
    print('get_fov output:', gproj_obj.get_fov())
    print('get_proj_plane_info:', gproj_obj.get_proj_plane_info())
    print('array_info:', gproj_obj.arrayinfo)

    
    # get projection map especially its value
    proj_m = gproj_obj.projmap(m, my_vec2pix_func)
    print('get_center output:', gproj_obj.get_center(lonlat=True))
    print(f'{proj_m=}')
    print(f'{proj_m.shape=}')

    # get position in the projection plane
    pos_xy = gproj_obj.ij2xy() # the output is a tuple
    print(f'{pos_xy=}')
    print(f'{pos_xy[0].shape=}')

    idx = gproj_obj.xy2ij(pos_xy)
    print(f'{idx=}')

    x,y = idxij = gproj_obj.ij2xy(i=10, j=20)
    print(f'{x=}, {y=}')


    plt.imshow(proj_m, extent=gproj_obj.get_extent(), origin='lower', cmap='viridis')
    plt.colorbar()
    plt.show()



