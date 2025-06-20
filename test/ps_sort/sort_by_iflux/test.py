import numpy as np
import healpy as hp

# 创建坐标转换器
rot = hp.rotator.Rotator(coord=['G', 'C'])

# 单个银经银纬点(l, b)，单位为度
l = 30.0  # 银经
b = 20.0  # 银纬

# 转换为赤经赤纬(RA, DEC)
ra, dec = rot(l, b)

print(f"银经银纬(l, b) = ({l:.2f}°, {b:.2f}°) -> 赤经赤纬(RA, DEC) = ({ra:.2f}°, {dec:.2f}°)")

