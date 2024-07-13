import numpy as np
import glob
import matplotlib.pyplot as plt


phi_real = np.arctan2(-250,500)
print(f'{phi_real=}')

phi_55 = glob.glob('./bias/5.5/phi_*.npy')
phi_55_list = []
for p in phi_55:
    P = np.load(p)
    phi_55_list.append(P)

phi_55_arr = np.array(phi_55_list)
phi_55_mean = np.mean(phi_55_arr)
print(f'{phi_55_mean=}')

phi_60 = glob.glob('./bias/6.0/phi_*.npy')
phi_60_list = []
for p in phi_60:
    P = np.load(p)
    phi_60_list.append(P)

phi_60_arr = np.array(phi_60_list)
phi_60_mean = np.mean(phi_60_arr)
print(f'{phi_60_mean=}')

phi_65 = glob.glob('./bias/6.5/phi_*.npy')
phi_65_list = []
for p in phi_65:
    P = np.load(p)
    phi_65_list.append(P)

phi_65_arr = np.array(phi_65_list)
phi_65_mean = np.mean(phi_65_arr)
print(f'{phi_65_mean=}')

phi_70 = glob.glob('./bias/7.0/phi_*.npy')
phi_70_list = []
for p in phi_70:
    P = np.load(p)
    phi_70_list.append(P)

phi_70_arr = np.array(phi_70_list)
phi_70_mean = np.mean(phi_70_arr)
print(f'{phi_70_mean=}')

phi_75 = glob.glob('./bias/7.5/phi_*.npy')
phi_75_list = []
for p in phi_75:
    P = np.load(p)
    phi_75_list.append(P)

phi_75_arr = np.array(phi_75_list)
phi_75_mean = np.mean(phi_75_arr)
print(f'{phi_75_mean=}')

phi_80 = glob.glob('./bias/8.0/phi_*.npy')
phi_80_list = []
for p in phi_80:
    P = np.load(p)
    phi_80_list.append(P)

phi_80_arr = np.array(phi_80_list)
phi_80_mean = np.mean(phi_80_arr)
print(f'{phi_80_mean=}')

phi_85 = glob.glob('./bias/8.5/phi_*.npy')
phi_85_list = []
for p in phi_85:
    P = np.load(p)
    phi_85_list.append(P)

phi_85_arr = np.array(phi_85_list)
phi_85_mean = np.mean(phi_85_arr)
print(f'{phi_85_mean=}')


phi_90 = glob.glob('./bias/9.0/phi_*.npy')
phi_90_list = []
for p in phi_90:
    P = np.load(p)
    phi_90_list.append(P)

phi_90_arr = np.array(phi_90_list)
phi_90_mean = np.mean(phi_90_arr)
print(f'{phi_90_mean=}')

phi_95 = glob.glob('./bias/9.5/phi_*.npy')
phi_95_list = []
for p in phi_95:
    P = np.load(p)
    phi_95_list.append(P)

phi_95_arr = np.array(phi_95_list)
phi_95_mean = np.mean(phi_95_arr)
print(f'{phi_95_mean=}')

phi_100 = glob.glob('./bias/10.0/phi_*.npy')
phi_100_list = []
for p in phi_100:
    P = np.load(p)
    phi_100_list.append(P)

phi_100_arr = np.array(phi_100_list)
phi_100_mean = np.mean(phi_100_arr)
print(f'{phi_100_mean=}')

phi_105 = glob.glob('./bias/10.5/phi_*.npy')
phi_105_list = []
for p in phi_105:
    P = np.load(p)
    phi_105_list.append(P)

phi_105_arr = np.array(phi_105_list)
phi_105_mean = np.mean(phi_105_arr)
print(f'{phi_105_mean=}')

phi_110 = glob.glob('./bias/11.0/phi_*.npy')
phi_110_list = []
for p in phi_110:
    P = np.load(p)
    phi_110_list.append(P)

phi_110_arr = np.array(phi_110_list)
phi_110_mean = np.mean(phi_110_arr)
print(f'{phi_110_mean=}')

phi_115 = glob.glob('./bias/11.5/phi_*.npy')
phi_115_list = []
for p in phi_115:
    P = np.load(p)
    phi_115_list.append(P)

phi_115_arr = np.array(phi_115_list)
phi_115_mean = np.mean(phi_115_arr)
print(f'{phi_115_mean=}')

phi_120 = glob.glob('./bias/12.0/phi_*.npy')
phi_120_list = []
for p in phi_120:
    P = np.load(p)
    phi_120_list.append(P)

phi_120_arr = np.array(phi_120_list)
phi_120_mean = np.mean(phi_120_arr)
print(f'{phi_120_mean=}')

phi_125 = glob.glob('./bias/12.5/phi_*.npy')
phi_125_list = []
for p in phi_125:
    P = np.load(p)
    phi_125_list.append(P)

phi_125_arr = np.array(phi_125_list)
phi_125_mean = np.mean(phi_125_arr)
print(f'{phi_125_mean=}')

phi_130 = glob.glob('./bias/13.0/phi_*.npy')
phi_130_list = []
for p in phi_130:
    P = np.load(p)
    phi_130_list.append(P)

phi_130_arr = np.array(phi_130_list)
phi_130_mean = np.mean(phi_130_arr)
print(f'{phi_130_mean=}')

x = [5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0]
y = [phi_55_mean, phi_60_mean, phi_65_mean, phi_70_mean, phi_75_mean, phi_80_mean, phi_85_mean, phi_90_mean, phi_95_mean, phi_100_mean, phi_105_mean, phi_110_mean, phi_115_mean, phi_120_mean, phi_125_mean, phi_130_mean]

plt.plot(x, [phi_real]*len(x), label=' constant ')

plt.plot(x, y, label=' different computation ')
plt.xlabel('beam size')
plt.ylabel('fitted polarization angle')
plt.show()

