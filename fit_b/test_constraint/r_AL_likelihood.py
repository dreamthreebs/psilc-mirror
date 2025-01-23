import numpy as np

cl_r = np.load(f'../')
cl_AL = np.load()

def load_data_inv_cov():
    _data_method = np.asarray([np.load(f'../nilc_5_freq/dl_res1/std/pcfn/{rlz_idx}.npy') for rlz_idx in range(1,200)])
    _data_noise = np.asarray([np.load(f'../nilc_5_freq/dl_res1/std/n_pcfn/{rlz_idx}.npy') for rlz_idx in range(1,200)])
    _data = _data_method - _data_noise
    _data_mean = np.mean(_data, axis=0)

    print(f'{_data.shape}')
    _cov = np.cov(_data_method, rowvar=False)
    _inv = np.linalg.inv(_cov)
    print(f'{_inv.shape=}')

    return _data_mean, _inv

def r_AL_likelihood():
    pass

    log_l = (_data_mean) @ _inv @ (_data_mean)
    print(f'{log_l=}')


load_data_inv_cov()

info = {"likelihood": {"r&AL": r_AL_likelihood}}
