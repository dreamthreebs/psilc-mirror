import numpy as np
import healpy as hp
import pickle
import matplotlib.pyplot as plt
from gnom_proj import GnomProj
import os



def gen_many_Cov_script():
    beam = 63 # arcmin
    lmax = 350
    nside = 2048
    ps_lon = 0
    ps_lat = 0
    m = np.load(f'./data/ps_maps/lon{ps_lon}lat{ps_lat}.npy')

    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)
    cl = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[:lmax+1,0]
    cl = cl * bl**2

    whole_width = 2.0 * beam
    size_arr = np.array([31, 41, 61, 81])
    reso_arr = whole_width / size_arr
    print(f'{reso_arr=}')
    if not os.path.exists('Cov_generator'):
        os.mkdir('Cov_generator')

    for size, reso in zip(size_arr, reso_arr):
        py_script_path = f"Cov_generator/Cov_size{size}reso{reso}.py"
        print(f'{py_script_path=}')
        my_string_py = f"""import numpy as np
import healpy as hp
import pickle
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("..")
from gnom_proj import GnomProj
os.chdir('..')

beam = 63 # arcmin
lmax = 350
nside = 2048
ps_lon = 0
ps_lat = 0
m = np.load(f'./data/ps_maps/lon{ps_lon}lat{ps_lat}.npy')

bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)
cl = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[:lmax+1,0]
cl = cl * bl**2

size = {size}
reso = {reso}
obj = GnomProj(m, lon=ps_lon, lat=ps_lat, xsize=size, ysize=size, reso=reso, nside=2048)
obj.print_init_info()
cov = obj.calc_cov(cl=cl, lmax=lmax)

np.save('./data/cov_size_{size}_reso{reso}.npy', cov)
"""
        with open(py_script_path, 'w') as py_script_file:
            py_script_file.write(my_string_py)

        my_string_sh = f'''#! /bin/bash
#SBATCH --partition=ali
#SBATCH --account=alicpt
#SBATCH --qos=regular
#SBATCH --job-name=wym
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6GB
#SBATCH --exclude=aliws[021-048],aliws005

#SBATCH -o /afs/ihep.ac.cn/users/w/wangyiming25/tmp/slurmlogs/o%j.log
#SBATCH --error=/afs/ihep.ac.cn/users/w/wangyiming25/tmp/slurmlogs/e%j.log

date +%m-%d_%H-%M
mpiexec python -u /afs/ihep.ac.cn/users/w/wangyiming25/work/dc2/psilc/test/proj_to_gnom/{py_script_path}
date +%m-%d_%H-%M
DATE=$(date +%m%d%H%M)
'''

        sh_script_path = f"./Cov_generator/Cov_size{size}reso{reso}.sh"
        with open(sh_script_path,'w') as sh_script_file:
            sh_script_file.write(my_string_sh)




def main():
    beam = 63 # arcmin
    lmax = 350
    nside = 2048
    ps_lon = 0
    ps_lat = 0
    m = np.load(f'./data/ps_maps/lon{ps_lon}lat{ps_lat}.npy')

    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)
    cl = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[:lmax+1,0]
    cl = cl * bl**2

    whole_width = 2.0 * beam
    size = 100
    reso = 1.7
    obj = GnomProj(m, lon=ps_lon, lat=ps_lat, xsize=size, ysize=size, reso=reso, nside=2048)
    obj.print_init_info()
    cov = obj.calc_cov(cl=cl, lmax=lmax)
    np.save(f'./data/cov_size_{size}_reso{reso}.npy', cov)
    # obj.test_flatten()


if __name__ == "__main__":
    gen_many_Cov_script()




