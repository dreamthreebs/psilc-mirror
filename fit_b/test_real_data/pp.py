import numpy as np

def check_num_ps_overlap():
    # num_ps = np.load('./num_ps.npy')
    overlap_ps = np.load('../../pp_P/215/overlap_ps.npy')
    # n_ps_3 = np.where(num_ps==3, num_ps, 0)
    # print(f'{n_ps_3=}')
    # idx_n_ps_3 = np.nonzero(n_ps_3)
    # print(f'{idx_n_ps_3=}')

    # print(f'{num_ps=}')
    print(f'{overlap_ps=}')



check_num_ps_overlap()

