import numpy as np

# 模拟 C 的 int32 溢出
def simulate_int32(value):
    int32_max = 2**31 - 1
    int32_min = -2**31
    return (value + 2**31) % (2**32) - 2**31

npix = 25000
i = 20000
kr = 1
kl = 1
j = 2

# 原始索引计算
index = (3 * i + kr) * 3 * npix + (3 * j + kl)

# 模拟溢出后索引
index_simulated = simulate_int32(index)

print("Original index:", index)
print("Simulated int32 index:", index_simulated)

# 使用模拟索引访问 NumPy 数组
arr = np.arange(2500000000)
try:
    print(arr[index])
    print(arr[index_simulated])
except IndexError as e:
    print(f"IndexError: {e}")

