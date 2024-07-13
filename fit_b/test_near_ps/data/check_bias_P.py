import numpy as np
import glob
import matplotlib.pyplot as plt

P_real = np.sqrt(250**2 + 500**2)
print(f'{P_real=}')

P_55 = glob.glob('./bias/5.5/P_*.npy')
P_55_list = []
for p in P_55:
    P = np.load(p)
    P_55_list.append(P)

P_55_arr = np.array(P_55_list)
P_55_mean = np.mean(P_55_arr)
print(f'{P_55_mean=}')

P_60 = glob.glob('./bias/6.0/P_*.npy')
P_60_list = []
for p in P_60:
    P = np.load(p)
    P_60_list.append(P)

P_60_arr = np.array(P_60_list)
P_60_mean = np.mean(P_60_arr)
print(f'{P_60_mean=}')

P_65 = glob.glob('./bias/6.5/P_*.npy')
P_65_list = []
for p in P_65:
    P = np.load(p)
    P_65_list.append(P)

P_65_arr = np.array(P_65_list)
P_65_mean = np.mean(P_65_arr)
print(f'{P_65_mean=}')

P_70 = glob.glob('./bias/7.0/P_*.npy')
P_70_list = []
for p in P_70:
    P = np.load(p)
    P_70_list.append(P)

P_70_arr = np.array(P_70_list)
P_70_mean = np.mean(P_70_arr)
print(f'{P_70_mean=}')

P_75 = glob.glob('./bias/7.5/P_*.npy')
P_75_list = []
for p in P_75:
    P = np.load(p)
    P_75_list.append(P)

P_75_arr = np.array(P_75_list)
P_75_mean = np.mean(P_75_arr)
print(f'{P_75_mean=}')

P_80 = glob.glob('./bias/8.0/P_*.npy')
P_80_list = []
for p in P_80:
    P = np.load(p)
    P_80_list.append(P)

P_80_arr = np.array(P_80_list)
P_80_mean = np.mean(P_80_arr)
print(f'{P_80_mean=}')

P_85 = glob.glob('./bias/8.5/P_*.npy')
P_85_list = []
for p in P_85:
    P = np.load(p)
    P_85_list.append(P)

P_85_arr = np.array(P_85_list)
P_85_mean = np.mean(P_85_arr)
print(f'{P_85_mean=}')


P_90 = glob.glob('./bias/9.0/P_*.npy')
P_90_list = []
for p in P_90:
    P = np.load(p)
    P_90_list.append(P)

P_90_arr = np.array(P_90_list)
P_90_mean = np.mean(P_90_arr)
print(f'{P_90_mean=}')

P_95 = glob.glob('./bias/9.5/P_*.npy')
P_95_list = []
for p in P_95:
    P = np.load(p)
    P_95_list.append(P)

P_95_arr = np.array(P_95_list)
P_95_mean = np.mean(P_95_arr)
print(f'{P_95_mean=}')

P_100 = glob.glob('./bias/10.0/P_*.npy')
P_100_list = []
for p in P_100:
    P = np.load(p)
    P_100_list.append(P)

P_100_arr = np.array(P_100_list)
P_100_mean = np.mean(P_100_arr)
print(f'{P_100_mean=}')

P_105 = glob.glob('./bias/10.5/P_*.npy')
P_105_list = []
for p in P_105:
    P = np.load(p)
    P_105_list.append(P)

P_105_arr = np.array(P_105_list)
P_105_mean = np.mean(P_105_arr)
print(f'{P_105_mean=}')

P_110 = glob.glob('./bias/11.0/P_*.npy')
P_110_list = []
for p in P_110:
    P = np.load(p)
    P_110_list.append(P)

P_110_arr = np.array(P_110_list)
P_110_mean = np.mean(P_110_arr)
print(f'{P_110_mean=}')

P_115 = glob.glob('./bias/11.5/P_*.npy')
P_115_list = []
for p in P_115:
    P = np.load(p)
    P_115_list.append(P)

P_115_arr = np.array(P_115_list)
P_115_mean = np.mean(P_115_arr)
print(f'{P_115_mean=}')

P_120 = glob.glob('./bias/12.0/P_*.npy')
P_120_list = []
for p in P_120:
    P = np.load(p)
    P_120_list.append(P)

P_120_arr = np.array(P_120_list)
P_120_mean = np.mean(P_120_arr)
print(f'{P_120_mean=}')

P_125 = glob.glob('./bias/12.5/P_*.npy')
P_125_list = []
for p in P_125:
    P = np.load(p)
    P_125_list.append(P)

P_125_arr = np.array(P_125_list)
P_125_mean = np.mean(P_125_arr)
print(f'{P_125_mean=}')

P_130 = glob.glob('./bias/13.0/P_*.npy')
P_130_list = []
for p in P_130:
    P = np.load(p)
    P_130_list.append(P)

P_130_arr = np.array(P_130_list)
P_130_mean = np.mean(P_130_arr)
print(f'{P_130_mean=}')

x = [5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0]
y = [P_55_mean, P_60_mean, P_65_mean, P_70_mean, P_75_mean, P_80_mean, P_85_mean, P_90_mean, P_95_mean, P_100_mean, P_105_mean, P_110_mean, P_115_mean, P_120_mean, P_125_mean, P_130_mean]

plt.plot(x, [P_real]*len(x), label=' constant ')

plt.plot(x, y, label=' different computation ')
plt.xlabel('beam size')
plt.ylabel('fitted flux density')
plt.show()

