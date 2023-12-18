import numpy as np
import pickle
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd

class GnomProj:
    def __init__(self, m:np.ndarray, lon:float, lat:float, xsize:int=40, ysize:int=40, reso:float=3.0, nside:int=2048):
        ''' Project spherical data into gnomonic space. Calculate the covariance matrix and try to fit point sources on that plane '''
        # np.set_printoptions(threshold=np.inf)
        self.lon = lon # degree
        self.lat = lat # degree
        self.xsize = xsize
        self.ysize = ysize
        self.reso = reso
        self.nside = nside
        self.m = m

        self.gproj_obj = hp.projector.GnomonicProj(rot=[self.lon, self.lat, 0], xsize=self.xsize, ysize=self.ysize, reso=self.reso)
        self.m_proj = self.gproj_obj.projmap(m, self.my_vec2pix_func)

        self.x,self.y = self.gproj_obj.ij2xy() # the x and y coordinate value of every point
        self.i,self.j = self.gproj_obj.xy2ij(x=self.x, y=self.y) # the i and j index of every point

    def my_vec2pix_func(self, x, y, z):
        return hp.vec2pix(nside=self.nside, x=x, y=y, z=z)

    def print_init_info(self):
        print('center position in degree:', self.gproj_obj.get_center(lonlat=True))
        print('center position in radians:', self.gproj_obj.get_center())
        print('extent in radians:', self.gproj_obj.get_extent())
        print('fov:', self.gproj_obj.get_fov())
        print('projection plane info:', self.gproj_obj.get_proj_plane_info())

        print(f'{self.m_proj=}')
        print(f'{self.m_proj.shape=}')

        print(f'{self.x=}, {self.y=}')
        print(f'{self.x.shape=}, {self.y.shape=}')

        print(f'{self.i=}, {self.j=}')
        print(f'{self.i.shape=}, {self.j.shape=}')

    def calc_C_theta(self, x, lmax, cl):
        ''' slow but accurate '''
        from numpy.polynomial.legendre import Legendre
        legendre_polys = [Legendre([0]*l + [1])(x) for l in range(lmax + 1)]
        coefficients = (2 * np.arange(lmax + 1) + 1) * cl
        sum_val = np.dot(coefficients, legendre_polys)
        return 1 / (4 * np.pi) * sum_val

    def calc_cov(self, cl, lmax=300):
        def evaluate_interp_func(l, x, interp_funcs):
            for interp_func, x_range in interp_funcs[l]:
                if x_range[0] <= x <= x_range[1]:
                    return interp_func(x)
            raise ValueError(f"x = {x} is out of the interpolation range for l = {l}")
        def calc_C_theta_itp(x, lmax, cl, itp_funcs):
            sum_val = 0.0
            for l in range(lmax + 1):
                sum_val += (2 * l + 1) * cl[l] * evaluate_interp_func(l, x, interp_funcs=itp_funcs)
            return 1/(4*np.pi)*sum_val

        n_pts = self.xsize * self.ysize
        cov = np.zeros((n_pts, n_pts))
        flatten_x = self.x.flatten()
        flatten_y = self.y.flatten()
        theta_cache = {}
        # cos_theta_list = [] # DEBUG:for checking if interpolate function could be used
        with open('../interpolate_cov/lgd_itp_funcs350.pkl', 'rb') as f:
            loaded_itp_funcs = pickle.load(f)

        for p1 in range(n_pts):
            print(f'{p1=}')
            for p2 in range(p1+1):
                if p1 == p2:
                    cov[p1,p2] = 1 / (4 * np.pi) * np.sum((2 * np.arange(lmax + 1) + 1) * cl[:lmax+1])
                else:
                    vec_p1 = self.gproj_obj.xy2vec(x=flatten_x[p1], y=flatten_y[p1] )
                    vec_p2 = self.gproj_obj.xy2vec(x=flatten_x[p2], y=flatten_y[p2] )
                    # print(f'{vec_p1=}')
                    cos_theta = np.dot(vec_p1, vec_p2)
                    cos_theta = min(1.0, max(cos_theta, -1.0))

                    if cos_theta not in theta_cache:

                        cov_p1p2 = calc_C_theta_itp(x=cos_theta, lmax=lmax, cl=cl[:lmax+1], itp_funcs=loaded_itp_funcs)
                        theta_cache[cos_theta] = cov_p1p2

                        # theta_cache[cos_theta] = 1;cos_theta_list.append(cos_theta) # DEBUG:for checking if interpolate function could be used(remember to comment the else function etc.)
                    else:
                        cov_p1p2 = theta_cache[cos_theta]

                    cov[p1, p2] = cov_p1p2
                    cov[p2, p1] = cov_p1p2
        return cov

    def fit_ps_ns_plane(self, beam):
        from iminuit import Minuit
        from iminuit.cost import LeastSquares

        sigma = np.deg2rad(beam) / 60 / np.sqrt(8 * np.log(2))

        def fit_model(theta, norm_beam, const):
            beam_profile = norm_beam / (2*np.pi*sigma**2) * np.exp(- (theta)**2 / (2 * sigma**2)) 
            # print(f'{beam_profile=}')
            return beam_profile + const

        n_pts = self.xsize * self.ysize
        flatten_x = self.x.flatten()
        flatten_y = self.y.flatten()

        np.set_printoptions(threshold=np.inf)
        theta = np.arctan(np.sqrt(flatten_x**2 + flatten_y**2))
        theta = np.nan_to_num(theta)
        print(f'{theta.shape=}')
        print(f'{theta=}')

        y = self.m_proj.flatten()
        print(f'{y.shape=}')

        y_err = 6.17 * np.ones_like(y)

        lsq = LeastSquares(x=theta, y=y, yerror=y_err, model=fit_model)
        obj_minuit = Minuit(lsq, norm_beam=1,  const=0)
        obj_minuit.limits = [(0,10),(-1e4,1e4)]
        # print(obj_minuit.scan(ncall=100))
        # obj_minuit.errors = (0.1, 0.2)
        print(obj_minuit.migrad())
        print(obj_minuit.hesse())

    def fit_ps_cmb_ns_plane(self, beam, nstd, cmbcov):
        from iminuit import Minuit
        from iminuit.cost import LeastSquares

        sigma = np.deg2rad(beam) / 60 / np.sqrt(8 * np.log(2))
        n_pts = self.xsize * self.ysize
        flatten_x = self.x.flatten()
        flatten_y = self.y.flatten()

        vec_center1 = hp.ang2vec(theta=self.lon, phi=self.lat, lonlat=True)
        print(f'{vec_center1=}')
        vec_center = self.gproj_obj.xy2vec(x=0, y=0)
        print(f'{vec_center=}')
        vec_around  = self.gproj_obj.xy2vec(x=flatten_x, y=flatten_y)

        theta = np.arccos(np.array(vec_center) @ np.array(vec_around))
        theta = np.nan_to_num(theta)
        print(f'{theta.shape=}')
        for i in range(n_pts):
            cmbcov[i,i] = cmbcov[i,i] + 6.17 **2
        inv_cov = np.linalg.inv(cmbcov) # now added noise

        def fit_model(theta, norm_beam, const):
            beam_profile = norm_beam / (2*np.pi*sigma**2) * np.exp(- (theta)**2 / (2 * sigma**2)) 
            # print(f'{beam_profile=}')
            return beam_profile + const


        def lsq(norm_beam, const):

            y_data = self.m_proj.flatten()
            print(f'{y_data.shape=}')
            y_model = fit_model(theta, norm_beam, const)
            y_diff = y_data - y_model
            z = (y_diff) @ inv_cov @ (y_diff)
            return z

        obj_minuit = Minuit(lsq, norm_beam=1,  const=0)
        obj_minuit.limits = [(0,10),(-1e4,1e4)]
        # print(obj_minuit.scan(ncall=100))
        # obj_minuit.errors = (0.1, 0.2)
        print(obj_minuit.migrad())
        print(obj_minuit.hesse())
        ndof = n_pts
        str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {ndof} = {obj_minuit.fval/ndof}"
        print(str_chi2)



    def see_map(self, xsize, ysize, reso):
        hp.gnomview(self.m, rot=[self.lon, self.lat, 0], xsize=xsize, ysize=ysize, reso=reso)
        plt.show()


    def test_flatten(self):
        i_arr = self.i.flatten()
        i2 = i_arr.reshape(300,200)
        i_diff = i2 - self.i
        np.set_printoptions(threshold=np.inf)
        print(f'{i_diff=}') # all is zero

    def test_origin(self):
        vec = self.gproj_obj.xy2vec(x=0, y=0)
        print(f'{vec=}')




if __name__ == '__main__':
    beam = 63 # arcmin
    lmax = 350
    df = pd.read_csv('../ps_sort/sort_by_iflux/40.csv')
    lon = df.at[44, 'lon']
    lat = df.at[44, 'lat']
    # iflux = df.at[44, 'iflux']
    # nstd = np.load('../../FGSim/NSTDNORTH/2048/40.npy')[0]
    # hp.mollview(nstd);plt.show()

    ps_lon = np.rad2deg(lon)
    ps_lat = np.rad2deg(lat)

    # ps_lon = 0
    # ps_lat = 0


    # m = np.load(f'./data/ps_maps/lon{ps_lon}lat{ps_lat}.npy')
    m = np.load(f'../../FGSim/PSNOISE/2048/40.npy')[0]
    # m = np.load(f'./data/ps_ns_maps/ps_ns.npy')
    # m = np.load(f'../../FGSim/STRPSCMBFGNOISE/40.npy')[0]
    hp.gnomview(m, rot=[ps_lon, ps_lat, 0])
    plt.show()
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/40.npy')[0]
    cmbcov = np.load('./data/cov_size_80_reso1.575.npy')

    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)
    cl = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[:lmax+1,0]
    cl = cl * bl**2

    xsize = 80
    ysize = 80
    # reso = 1.0 * hp.nside2resol(nside=2048, arcmin=True)
    reso = 1.575
    obj = GnomProj(m, lon=ps_lon, lat=ps_lat, xsize=xsize, ysize=ysize, reso=reso)
    obj.print_init_info()

    # cov = obj.calc_cov(cl=cl, lmax=lmax)
    # np.save(f'./data/cov_lon{ps_lon}lat{ps_lat}.npy', cov)
    # obj.test_flatten()

    # obj.see_map(xsize=xsize, ysize=ysize, reso=reso)

    obj.test_origin()
    obj.fit_ps_ns_plane(beam=beam)
    # obj.fit_ps_cmb_ns_plane(beam=beam, nstd=nstd, cmbcov=cmbcov)


