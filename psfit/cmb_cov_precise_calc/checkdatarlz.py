import numpy as np

data_list = []
for i in range(2000):
    print(f'{i=}')
    m = np.load(f'./datarlz/{i}.npy')
    print(f'{m=}')
    # print(f"{np.count_nonzero(m)=}")
    data_list.append(m)

data = np.array(data_list)
print(f'{data.shape=}')
np.save('data2k.npy', data)

