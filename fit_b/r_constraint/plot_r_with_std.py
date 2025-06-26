import numpy as np
import matplotlib.pyplot as plt

method_list = ['FG + CMB + NOISE', 'PS + FG + CMB + NOISE', 'Template fitting', 'Recycling + Inpaint on B', 'Masking on B']
r_mean_list = [ 1.89e-3, 3.56e-3, 2.18e-3, 1.05e-2, 3.08e-3]
r_std_list = [2.40e-3, 2.52e-3, 2.45e-3, 3.01e-3, 2.59e-3]
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
plt.savefig('/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20250609/r_constraints.png', dpi=300)
plt.show()

