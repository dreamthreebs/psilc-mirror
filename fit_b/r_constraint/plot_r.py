import numpy as np
import matplotlib.pyplot as plt

method_list = ['PS + FG + CMB + NOISE', 'FG + CMB + NOISE', 'Template fitting', 'Recycling + Inpaint on B']
r_mean_list = [2.57e-3, 6.94e-4, 1.20e-3, 9.92e-3]
r_std_list = [2.61e-3, 2.54e-3, 2.60e-3, 3.28e-3]
r_std_list = [x*2 for x in r_std_list]

fig, ax = plt.subplots()
for i, (method, r_mean, r_std) in enumerate(zip(method_list, r_mean_list, r_std_list)):
    ax.errorbar(i, r_mean, yerr=r_std, fmt='.', label=method, capsize=5)
ax.set_xticks(range(len(method_list)))
ax.set_yscale("log")
# ax.set_xticklabels(method_list)
ax.set_xlabel('different methods')
ax.set_ylabel('tensor to scalar ratio r')
# ax.set_xticklabels(method_list, rotation=45, ha="left")

ax.legend()
plt.tight_layout()
plt.savefig('/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20250131/r_constraints.png', dpi=300)
plt.show()

