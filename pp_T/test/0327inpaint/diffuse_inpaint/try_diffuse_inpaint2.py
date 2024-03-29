import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

def process_source(idx_src, src_idx, nside, beam, m):
    print(f'Processing {idx_src=}')

    src_lon, src_lat = hp.pix2ang(nside=nside, ipix=src_idx[idx_src], lonlat=True)
    ipix_disc = hp.query_disc(nside=nside, vec=hp.pix2vec(nside=nside, ipix=src_idx[idx_src]), radius=1.5*np.deg2rad(beam)/60)
    
    neighbour_ipix = hp.get_all_neighbours(nside=nside, theta=ipix_disc)

    for idx_itr in np.arange(2000):
        for idx_disc in np.arange(neighbour_ipix.shape[1]):
            m[ipix_disc[idx_disc]] = np.mean(m[neighbour_ipix[:,idx_disc]])

    return m

def main():
    nside = 2048
    beam = 17

    mask = np.load('../mask.npy')
    ps_cmb_noise = np.load('../ps_cmb_noise.npy')
    m = ps_cmb_noise * mask

    src_idx = np.load('../source_indices.npy')
    print(f'{src_idx=}')

    # 并行处理
    with ProcessPoolExecutor() as executor:
        futures = []
        for idx_src in np.arange(len(src_idx)):
            # 提交处理每个源点的任务
            # 注意：这里为了避免多进程间共享大数组`m`引起的问题，可能需要特别的设计
            # 可能的解决方案是返回修改的索引和值，然后在主进程中更新`m`
            future = executor.submit(process_source, idx_src, src_idx, nside, beam, m)
            futures.append(future)

        # 收集结果
        for future in futures:
            # 这里应用每个进程返回的修改
            # 注意：需要根据实际返回值调整此处的逻辑
            m = future.result()

    np.save('diffuse_m.npy', m)
    hp.mollview(m, xsize=4000)
    plt.show()

if __name__ == "__main__":
    main()

