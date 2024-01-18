import numpy as np

cov = np.load('../../fitv3/cov/1.npy')

# A = np.random.rand(100, 100)
# cov = np.dot(A, A.transpose())

inv_mat1 = np.linalg.inv(cov)
inv_mat2,_,_,_ = np.linalg.lstsq(cov, np.eye(cov.shape[0]),rcond=None)

# cov1 = np.load('./172/1.npy')
# cov2 = np.load('./180/1.npy')
# cov3 = np.load('./190/1.npy')

print(f'{cov=}')


eigenvalues_min = np.min(np.linalg.eigvalsh(cov))
print(f'{eigenvalues_min=}')
epsilon = 1e-8
cov = cov + epsilon * np.eye(cov.shape[0])

cond_number = np.linalg.cond(cov)
print("Condition number of the matrix:", cond_number)



# print(f'{cov1=}')
# print(f'{cov2=}')
# print(f'{cov3=}')

def is_symmetric_positive_semidefinite(matrix):
    # 检查矩阵是否对称
    if not np.allclose(matrix, matrix.T):
        print('not symmetric')
        return False
    
    # 计算特征值
    eigenvalues = np.linalg.eigvalsh(matrix)
    np.set_printoptions(threshold=np.inf)
    print(f'{eigenvalues=}')
    
    # 检查特征值是否都是非负的
    if np.all(eigenvalues >= 0):
        return True
    else:
        print('no!!!!!!!!!')
        return False

# 示例
print("Is symmetric positive semi-definite:", is_symmetric_positive_semidefinite(cov))



