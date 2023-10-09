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

corrected = np.load(f'./0/cleaned_B.npy')
corrupted = np.load(f'./0/corrupted_B.npy')
cl_corrupted = hp.anafast(corrupted, lmax=lmax)
cl_corrected = hp.anafast(corrected, lmax=lmax)
plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky0, label=f'bin corrupted', color='black')
plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky0, label=f'bin corrected QU', linestyle='--', color='black')

corrected = np.load(f'./1/cleaned_B.npy')
corrupted = np.load(f'./1/corrupted_B.npy')
cl_corrupted = hp.anafast(corrupted, lmax=lmax)
cl_corrected = hp.anafast(corrected, lmax=lmax)
# plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky0, label=f'bin corrupted B')
plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky0, label=f'bin corrected B', linestyle=':', color='black')


# corrected = np.load(f'./2/cleaned_B.npy')
# corrupted = np.load(f'./2/corrupted_B.npy')
# cl_corrupted = hp.anafast(corrupted, lmax=lmax)
# cl_corrected = hp.anafast(corrected, lmax=lmax)
# plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky1, label=f'C2 5 corrupted QU')
# plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky1, label=f'C2 5 corrected QU')

# corrected = np.load(f'./3/cleaned_B.npy')
# corrupted = np.load(f'./3/corrupted_B.npy')
# cl_corrupted = hp.anafast(corrupted, lmax=lmax)
# cl_corrected = hp.anafast(corrected, lmax=lmax)
# plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky1, label=f'C2 5 corrupted B')
# plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky1, label=f'C2 5 corrected B')

# corrected = np.load(f'./4/cleaned_B.npy')
# corrupted = np.load(f'./4/corrupted_B.npy')
# cl_corrupted = hp.anafast(corrupted, lmax=lmax)
# cl_corrected = hp.anafast(corrected, lmax=lmax)
# plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky2, label=f'C2 10 corrupted QU')
# plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky2, label=f'C2 10 corrected QU')

# corrected = np.load(f'./5/cleaned_B.npy')
# corrupted = np.load(f'./5/corrupted_B.npy')
# cl_corrupted = hp.anafast(corrupted, lmax=lmax)
# cl_corrected = hp.anafast(corrected, lmax=lmax)
# plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky2, label=f'C2 10 corrupted B')
# plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky2, label=f'C2 10 corrected B')


# corrected = np.load(f'./6/cleaned_B.npy')
# corrupted = np.load(f'./6/corrupted_B.npy')
# cl_corrupted = hp.anafast(corrupted, lmax=lmax)
# cl_corrected = hp.anafast(corrected, lmax=lmax)
# plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky3, label=f'C1 12 corrupted QU')
# plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky3, label=f'C1 12 corrected QU')

# corrected = np.load(f'./7/cleaned_B.npy')
# corrupted = np.load(f'./7/corrupted_B.npy')
# cl_corrupted = hp.anafast(corrupted, lmax=lmax)
# cl_corrected = hp.anafast(corrected, lmax=lmax)
# plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky3, label=f'C1 12 corrupted B')
# plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky3, label=f'C1 12 corrected B')

# corrected = np.load(f'./8/cleaned_B.npy')
# corrupted = np.load(f'./8/corrupted_B.npy')
# cl_corrupted = hp.anafast(corrupted, lmax=lmax)
# cl_corrected = hp.anafast(corrected, lmax=lmax)
# plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky4, label=f'C1 13 corrupted QU')
# plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky4, label=f'C1 13 corrected QU')

# corrected = np.load(f'./9/cleaned_B.npy')
# corrupted = np.load(f'./9/corrupted_B.npy')
# cl_corrupted = hp.anafast(corrupted, lmax=lmax)
# cl_corrected = hp.anafast(corrected, lmax=lmax)
# plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky4, label=f'C1 13 corrupted B')
# plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky4, label=f'C1 13 corrected B')

# corrected = np.load(f'./10/cleaned_B.npy')
# corrupted = np.load(f'./10/corrupted_B.npy')
# cl_corrupted = hp.anafast(corrupted, lmax=lmax)
# cl_corrected = hp.anafast(corrected, lmax=lmax)
# plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky5, label=f'C1 20 corrupted QU')
# plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky5, label=f'C1 20 corrected QU')

# corrected = np.load(f'./11/cleaned_B.npy')
# corrupted = np.load(f'./11/corrupted_B.npy')
# cl_corrupted = hp.anafast(corrupted, lmax=lmax)
# cl_corrected = hp.anafast(corrected, lmax=lmax)
# plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky5, label=f'C1 20 corrupted B')
# plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky5, label=f'C1 20 corrected B')

# corrected = np.load(f'./14/cleaned_B.npy')
# corrupted = np.load(f'./14/corrupted_B.npy')
# cl_corrupted = hp.anafast(corrupted, lmax=lmax)
# cl_corrected = hp.anafast(corrected, lmax=lmax)
# plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky6, label=f'C1 25 corrupted QU')
# plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky6, label=f'C1 25 corrected QU')

# corrected = np.load(f'./15/cleaned_B.npy')
# corrupted = np.load(f'./15/corrupted_B.npy')
# cl_corrupted = hp.anafast(corrupted, lmax=lmax)
# cl_corrected = hp.anafast(corrected, lmax=lmax)
# # plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky6, label=f'C1 25 corrupted B')
# plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky6, label=f'C1 25 corrected B')

# corrected = np.load(f'./16/cleaned_B.npy')
# corrupted = np.load(f'./16/corrupted_B.npy')
# cl_corrupted = hp.anafast(corrupted, lmax=lmax)
# cl_corrected = hp.anafast(corrected, lmax=lmax)
# plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky4, label=f'C1 13 corrupted QU')
# plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky4, label=f'C1 13 corrected QU')

# corrected = np.load(f'./17/cleaned_B.npy')
# corrupted = np.load(f'./17/corrupted_B.npy')
# cl_corrupted = hp.anafast(corrupted, lmax=lmax)
# cl_corrected = hp.anafast(corrected, lmax=lmax)
# # plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky4, label=f'C1 13 corrupted B')
# plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky4, label=f'C1 13 corrected B')

# corrected = np.load(f'./18/cleaned_B.npy')
# corrupted = np.load(f'./18/corrupted_B.npy')
# cl_corrupted = hp.anafast(corrupted, lmax=lmax)
# cl_corrected = hp.anafast(corrected, lmax=lmax)
# plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky1, label=f'C1 5 corrupted nside 2048', color='purple')
# plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky1, label=f'C1 5 corrected QU nside 2048', color='purple', linestyle='--')

# corrected = np.load(f'./19/cleaned_B.npy')
# corrupted = np.load(f'./19/corrupted_B.npy')
# cl_corrupted = hp.anafast(corrupted, lmax=lmax)
# cl_corrected = hp.anafast(corrected, lmax=lmax)
# # plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky1, label=f'C1 5 corrupted B')
# plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky1, label=f'C1 5 corrected B nside 2048', color='purple', linestyle=':')

# corrected = np.load(f'./20/cleaned_B.npy')
# corrupted = np.load(f'./21/corrupted_B.npy')
# cl_corrupted = hp.anafast(corrupted, lmax=lmax)
# cl_corrected = hp.anafast(corrected, lmax=lmax)
# plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky8, label=f'C1 5 corrupted nside 512', color='green')
# plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky8, label=f'C1 5 corrected QU nside 512', linestyle='--', color='green')

# corrected = np.load(f'./21/cleaned_B.npy')
# corrupted = np.load(f'./21/corrupted_B.npy')
# cl_corrupted = hp.anafast(corrupted, lmax=lmax)
# cl_corrected = hp.anafast(corrected, lmax=lmax)
# # plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky8, label=f'C1 5 corrupted B nside 512')
# plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky8, label=f'C1 5 corrected B nside 512', linestyle=':', color='green')

corrected = np.load(f'./22/cleaned_B.npy')
corrupted = np.load(f'./22/corrupted_B.npy')
cl_corrupted = hp.anafast(corrupted, lmax=lmax)
cl_corrected = hp.anafast(corrected, lmax=lmax)
plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky7, label=f'corrupted QU')
plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky7, label=f'corrected QU')

corrected = np.load(f'./23/cleaned_B.npy')
corrupted = np.load(f'./23/corrupted_B.npy')
cl_corrupted = hp.anafast(corrupted, lmax=lmax)
cl_corrected = hp.anafast(corrected, lmax=lmax)
plt.loglog(l*(l+1)*cl_corrupted/(2*np.pi)/fsky7, label=f'corrupted B', color='purple')
plt.loglog(l*(l+1)*cl_corrected/(2*np.pi)/fsky7, label=f'corrected B', color='purple', linestyle='--')




plt.xlabel('$\\ell$')
plt.ylabel('$D_\\ell$')
# plt.xlim(2, 300)
plt.ylim(1e-15, 1e0)

plt.legend()
plt.show()




