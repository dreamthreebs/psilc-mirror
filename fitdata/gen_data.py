import numpy as np
import healpy as hp
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class GenerateData:
    def __init__(self, n_rlz_begin, n_rlz_end, n_rlz=100):
        self.n_rlz_begin = n_rlz_begin
        self.n_rlz_end = n_rlz_end
        self.n_rlz = n_rlz


    def gen_CMB(self, freq:'GHz', beam:'arcmin'):
        print(f'begin gen CMB ...')

        path_cmb = Path(f'./2048/CMB/{freq}')
        path_cmb.mkdir(parents=True, exist_ok=True)

        for i in range(self.n_rlz_begin, self.n_rlz_end):
            print(f'{i=}')
            m = np.load(f'../src/cmbsim/cmbdata/m_realization/{i}.npy')
            sm = hp.smoothing(m, fwhm=np.deg2rad(beam) / 60, lmax=2000)
            np.save(f'./2048/CMB/{freq}/{i}.npy', sm)

    def gen_PS(self, freq:'GHz'):
        print(f'begin gen strong PS ...')

        if not os.path.exists('./2048/PS'):
            os.mkdir('./2048/PS')

        if not os.path.exists(f'./2048/PS/{freq}'):
            os.mkdir(f'./2048/PS/{freq}')

        ps = hp.read_map(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/observations/AliCPT_uKCMB/{freq}GHz/group3_map_{freq}GHz.fits', field=(0,1,2))
        np.save(f'./2048/PS/{freq}/ps.npy', ps)

    def gen_FG(self, freq:'GHz'):
        print(f'begin gen diffuse foreground ...')

        if not os.path.exists('./2048/FG'):
            os.mkdir('./2048/FG')

        if not os.path.exists(f'./2048/FG/{freq}'):
            os.mkdir(f'./2048/FG/{freq}')

        fg = hp.read_map(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/observations/AliCPT_uKCMB/{freq}GHz/group1_map_{freq}GHz.fits', field=(0,1,2))
        np.save(f'./2048/FG/{freq}/fg.npy', fg)

    def gen_NOISE(self, freq:'GHz'):
        print(f'begin gen noise ...')

        path_noise = Path(f'./2048/NOISE/{freq}')
        path_noise.mkdir(parents=True, exist_ok=True)

        nstd = np.load(f'../FGSim/NSTDNORTH/2048/{freq}.npy')

        for i in range(self.n_rlz_begin, self.n_rlz_end):
            print(f'{i=}')
            noise = nstd * np.random.normal(0, 1, size=(nstd.shape[0],nstd.shape[1]))
            np.save(f'./2048/NOISE/{freq}/{i}.npy', noise)

    def gen_NOISE_LOW(self, freq:'GHz'):
        print(f'begin gen noise ...')

        path_noise = Path(f'./2048/NOISE_LOW/{freq}')
        path_noise.mkdir(parents=True, exist_ok=True)

        nstd = np.load(f'../FGSim/NSTDNORTH/2048/{freq}.npy')

        for i in range(self.n_rlz_begin, self.n_rlz_end):
            print(f'{i=}')
            noise = nstd / 10 * np.random.normal(0, 1, size=(nstd.shape[0],nstd.shape[1]))
            np.save(f'./2048/NOISE_LOW/{freq}/{i}.npy', noise)

    def downgrade_inside(self, directory:str, nside_out:int, freq:'GHz'):
        save_path = os.path.join(f'./{nside_out}', directory)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(os.path.join(save_path, f'{freq}')):
            os.makedirs(os.path.join(save_path, f'{freq}'))

        for i in range(self.n_rlz):
            print(f'{i=}')
            m = np.load(os.path.join('./2048', directory, f'{freq}', f'{i}.npy'))
            ud_m = hp.ud_grade(m, nside_out=nside_out)

            np.save(os.path.join(os.path.join(save_path, f'{freq}'),f'{i}.npy'), ud_m)

    def downgrade_one_map(self, directory:str, nside_out:int, freq:'GHz'):
        map_type = directory.lower()

        save_path = os.path.join(f'./{nside_out}', directory)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(os.path.join(save_path, f'{freq}')):
            os.makedirs(os.path.join(save_path, f'{freq}'))

        m = np.load(os.path.join('./2048', directory, f'{freq}', f'{map_type}.npy'))
        ud_m = hp.ud_grade(m, nside_out=nside_out)

        np.save(os.path.join(os.path.join(save_path, f'{freq}'),f'{map_type}.npy'), ud_m)

    def check_some_map(self):
        m = np.load('./1024/PS/40/ps.npy')[0]
        # m = np.load('../FGSim/NSTDNORTH/2048/40.npy')[0]
        print(f'{np.std(m)=}')
        # hp.mollview(m)
        hp.gnomview(m)
        plt.show()

if __name__ == "__main__":
    n_rlz = 50
    df = pd.read_csv('../FGSim/FreqBand')

    freq = df.at[7, "freq"]
    beam = df.at[7, "beam"]
    print(f'{freq=}, {beam=}')
    n_rlz_begin = 0
    n_rlz_end = 100
    obj = GenerateData(n_rlz_begin=n_rlz_begin, n_rlz_end=n_rlz_end)
    # obj.gen_PS(freq=freq)
    # obj.gen_FG(freq=freq)
    # obj.gen_CMB(freq=freq, beam=beam)
    # obj.gen_NOISE(freq=freq)
    obj.gen_NOISE_LOW(freq=freq)

    # obj.downgrade_inside(directory='NOISE', nside_out=256, freq=freq)
    # obj.downgrade_one_map(directory='PS', nside_out=1024, freq=freq)

    # obj.check_some_map()

