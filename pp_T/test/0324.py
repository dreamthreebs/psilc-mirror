import numpy as np
from iminuit import Minuit

# 生成示例数据
np.random.seed(42)
x = np.linspace(0, 10, 20)
true_a, true_b = 2.5, -0.3
y = true_a * x + true_b + np.random.normal(0, 1, x.size)  # 简单线性关系加噪声

# 假设的数据协方差矩阵（实际应用中应根据数据估计得到）
# 这里简化为对角矩阵，每个元素代表对应数据点误差的方差
cov_matrix = np.diag(np.linspace(0.1, 0.5, x.size)**2)

# 普通最小二乘损失函数
def least_squares(a, b):
    residuals = y - (a * x + b)
    return np.sum(residuals**2)

# 广义最小二乘损失函数，考虑协方差矩阵
def generalized_least_squares(a, b):
    residuals = y - (a * x + b)
    # 利用协方差矩阵的逆进行加权
    return residuals.T @ np.linalg.inv(cov_matrix) @ residuals

# 普通最小二乘拟合
m_ols = Minuit(least_squares, a=1, b=0)
m_ols.migrad()  # 运行拟合

# 广义最小二乘拟合
m_gls = Minuit(generalized_least_squares, a=1, b=0)
m_gls.migrad()

print(m_ols.values, m_gls.values)  # 打印两种方法的拟合结果
