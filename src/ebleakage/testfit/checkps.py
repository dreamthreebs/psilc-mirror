import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

lmax = 2000
l = np.arange(lmax+1)

mask0 = np.load('../circle_mask.npy')
mask1 = np.load('../apo_circle_mask2048.npy')
mask2 = np.load('../apo_circle_mask2048C2_10.npy')
mask3 = np.load('../apo_circle_mask2048C1_12.npy')
mask4 = np.load('../apo_circle_mask2048C1_13.npy')
mask5 = np.load('../apo_circle_mask2048C1_20.npy')
mask6 = np.load('../apo_circle_mask2048C1_25.npy')
mask7 = np.load('../circle_mask.npy')
mask8 = np.load('../apo_circle_mask.npy')

fsky0 = np.sum(mask0) / np.size(mask0)
fsky1 = np.sum(mask1) / np.size(mask1)
fsky2 = np.sum(mask2) / np.size(mask2)
fsky3 = np.sum(mask3) / np.size(mask3)
fsky4 = np.sum(mask4) / np.size(mask4)
fsky5 = np.sum(mask5) / np.size(mask5)
fsky6 = np.sum(mask6) / np.size(mask6)
fsky7 = np.sum(mask7) / np.size(mask7)
fsky8 = np.sum(mask8) / np.size(mask8)

# corrected = np.load(f'./0/cleaned_B.npy')
# corrupted = np.load(f'./0/corrupted_B.npy')
# cl_corrupted = hp.anafast(corrupted, lmax=lmax)
# cl_corrected = hp.anafast(corrected, lmax=lmax)
# plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky0, label=f'bin corrupted QU')
# plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky0, label=f'bin corrected QU', linestyle='--')

# corrected = np.load(f'./1/cleaned_B.npy')
# corrupted = np.load(f'./1/corrupted_B.npy')
# cl_corrupted = hp.anafast(corrupted, lmax=lmax)
# cl_corrected = hp.anafast(corrected, lmax=lmax)
# plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky0, label=f'bin corrupted B')
# plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky0, label=f'bin corrected B', linestyle=':')

# corrected = np.load(f'./3/cleaned_B.npy')
# corrupted = np.load(f'./3/corrupted_B.npy')
# cl_corrupted = hp.anafast(corrupted, lmax=lmax)
# cl_corrected = hp.anafast(corrected, lmax=lmax)
# plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky1, label=f'apo corrupted QU ')
# plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky1, label=f'apo corrected QU ', linestyle=':')

# corrected = np.load(f'./4/cleaned_B.npy')
# corrupted = np.load(f'./4/corrupted_B.npy')
# cl_corrupted = hp.anafast(corrupted, lmax=lmax)
# cl_corrected = hp.anafast(corrected, lmax=lmax)
# plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky1, label=f'apo corrupted QU B')
# plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky1, label=f'apo corrected QU B', linestyle=':')

# corrected = np.load(f'./5/cleaned_B.npy')
# corrupted = np.load(f'./5/corrupted_B.npy')
# cl_corrupted = hp.anafast(corrupted, lmax=lmax)
# cl_corrected = hp.anafast(corrected, lmax=lmax)
# plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky1, label=f'apo apo corrupted QU')
# plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky1, label=f'apo apo corrected QU', linestyle=':')

corrected = np.load(f'./6/cleaned_B.npy')
corrupted = np.load(f'./6/corrupted_B.npy')
cl_corrupted = hp.anafast(corrupted, lmax=lmax)
cl_corrected = hp.anafast(corrected, lmax=lmax)
plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky0, label=f'bin corrupted ')
plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky0, label=f'bin corrected fit QU', linestyle='--')

corrected = np.load(f'./7/cleaned_B.npy')
corrupted = np.load(f'./7/corrupted_B.npy')
cl_corrupted = hp.anafast(corrupted, lmax=lmax)
cl_corrected = hp.anafast(corrected, lmax=lmax)
# plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky0, label=f'bin corrupted fit B from QU')
plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky0, label=f'bin corrected fit B from QU', linestyle=':')

corrected = np.load(f'./8/cleaned_B.npy')
corrupted = np.load(f'./8/corrupted_B.npy')
cl_corrupted = hp.anafast(corrupted, lmax=lmax)
cl_corrected = hp.anafast(corrected, lmax=lmax)
# plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky0, label=f'bin corrupted fit B')
plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky0, label=f'bin corrected fit B', linestyle=':')

corrected = np.load(f'./9/cleaned_B.npy')
corrupted = np.load(f'./9/corrupted_B.npy')
cl_corrupted = hp.anafast(corrupted, lmax=lmax)
cl_corrected = hp.anafast(corrected, lmax=lmax)
plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky1, label=f'apo C1 5 corrupted fit QU')
plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky1, label=f'apo C1 5 corrected fit QU', linestyle=':')

corrected = np.load(f'./10/cleaned_B.npy')
corrupted = np.load(f'./10/corrupted_B.npy')
cl_corrupted = hp.anafast(corrupted, lmax=lmax)
cl_corrected = hp.anafast(corrected, lmax=lmax)
# plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky1, label=f'apo corrupted fit QU')
plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky1, label=f'apo C1 5 corrected fit B', linestyle='--')

corrected = np.load(f'./11/cleaned_B.npy')
corrupted = np.load(f'./11/corrupted_B.npy')
cl_corrupted = hp.anafast(corrupted, lmax=lmax)
cl_corrected = hp.anafast(corrected, lmax=lmax)
plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky2, label=f'apo C2 10 corrupted fit QU')
plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky2, label=f'apo C2 10 corrected fit QU', linestyle=':')

corrected = np.load(f'./12/cleaned_B.npy')
corrupted = np.load(f'./12/corrupted_B.npy')
cl_corrupted = hp.anafast(corrupted, lmax=lmax)
cl_corrected = hp.anafast(corrected, lmax=lmax)
# plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky1, label=f'apo corrupted fit QU')
plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky2, label=f'apo C2 10 corrected fit B', linestyle='--')

corrected = np.load(f'./13/cleaned_B.npy')
corrupted = np.load(f'./13/corrupted_B.npy')
cl_corrupted = hp.anafast(corrupted, lmax=lmax)
cl_corrected = hp.anafast(corrected, lmax=lmax)
plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky5, label=f'apo C1 20 corrupted fit QU')
plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky5, label=f'apo C1 20 corrected fit QU', linestyle=':')

corrected = np.load(f'./14/cleaned_B.npy')
corrupted = np.load(f'./14/corrupted_B.npy')
cl_corrupted = hp.anafast(corrupted, lmax=lmax)
cl_corrected = hp.anafast(corrected, lmax=lmax)
# plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky5, label=f'apo corrupted fit QU')
plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky5, label=f'apo C1 20 corrected fit B', linestyle='--')



plt.xlabel('$\\ell$')
plt.ylabel('$D_\\ell$')
# plt.xlim(2, 1500)
plt.ylim(1e-15, 1e0)

plt.legend()
plt.show()




