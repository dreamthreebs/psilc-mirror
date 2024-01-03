import numpy as np

def alternate_loop_with_index(array):
    n = len(array)
    mid = n // 2  # 找到中间索引
    yield mid, array[mid]

    left, right = mid - 1, mid + 1

    while left >= 0 or right < n:
        if right < n:
            yield right, array[right]
            right += 1

        if left >= 0:
            yield left, array[left]
            left -= 1

# 被循环的数组
array = [1, 2, 3, 4, 5, 6]
array = np.arange(1900)

# 使用定义的函数进行循环
for index, value in alternate_loop_with_index(array):
    print(f"Index: {index}, Value: {value}")

