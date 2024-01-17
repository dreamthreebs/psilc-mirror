import numpy as np

def is_symmetric_positive_semidefinite(matrix):
    # 检查矩阵是否对称
    if not np.allclose(matrix, matrix.T):
        print('not symmetric')
        return False

    # 计算特征值
    eigenvalues = np.linalg.eigvalsh(matrix)
    # np.set_printoptions(threshold=np.inf)
    # print(f'{eigenvalues=}')

    # 检查特征值是否都是非负的
    if np.all(eigenvalues >= 0):
        return True
    else:
        print('no!!!!!!!!!')
        return False

def check_eigen_val():
    for i in range(19,135):
        print(f'{i = }')
    
        cov = np.load(f'./cov/{i}.npy')
        # print("Is origin cov symmetric positive semi-definite:", is_symmetric_positive_semidefinite(cov))
        eigen_val = np.linalg.eigvalsh(cov)
        min_eigin_val = np.min(eigen_val)
        print(f'the smallest eigen_val is:{min_eigin_val}')
    
        # epsilon = 1e-4
        # if epsilon < np.abs(min_eigin_val):
        #     raise ValueError(f"epsilon = {epsilon} is not enough")
    
        # regular_cov = cov + epsilon * np.eye(cov.shape[0])
        # is_right_cov = is_symmetric_positive_semidefinite(regular_cov)
        # if is_right_cov:
        #     print("Is regular cov symmetric positive semi-definite:", is_right_cov)
        # else:
        #     print('not right cov, please check')

def check_cov():
    cov = np.load(f'./cov/1.npy')
    # np.set_printoptions(threshold=np.inf)
    print(f'{cov[0:30,0:30]=}')

check_cov()

