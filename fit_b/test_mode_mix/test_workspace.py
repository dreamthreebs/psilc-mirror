import numpy as np
import healpy as hp
import pymaster as nmt
import matplotlib.pyplot as plt

from pathlib import Path

nside = 256
fg_seeds = np.load('../seeds_fg_2k.npy')
beam = 0
rlz_idx=0
ls = np.arange(3*nside)

bin_mask = np.load('../../src/mask/north/BINMASKG1024.npy')
# apo_mask = nmt.mask_apodization(mask_in=bin_mask, aposize=5)
apo_mask = hp.ud_grade(nmt.mask_apodization(bin_mask, 5.0, "C1"), nside_out=nside)


cls_fg = np.load('./data/cl_fg/cl_fg_10010.npy')
cl_aa = cls_fg[0,:3*nside]
cl_ee = cls_fg[1,:3*nside]
cl_bb = cls_fg[2,:3*nside]
cl_ae = np.zeros(3*nside)
def gen_sim():
    # np.random.seed(seed=fg_seeds[rlz_idx])
    fg_rlz = hp.synfast(cls_fg, nside=nside, new=True, fwhm=np.deg2rad(beam)/60)
    return fg_rlz

mpa, mpQ, mpU = gen_sim()

# hp.mollview(mpI, norm='symlog2')
# hp.mollview(mpQ)
# hp.mollview(mpU)
# plt.show()

f0 = nmt.NmtField(apo_mask, [mpa])
f2 = nmt.NmtField(apo_mask, [mpQ, mpU])

pcl_aa = nmt.compute_coupled_cell(f0, f0)[0]
pcl_aE, pcl_aB = nmt.compute_coupled_cell(f0, f2)
pcl_EE, pcl_EB, pcl_BE, pcl_BB = nmt.compute_coupled_cell(f2, f2)
fsky = np.mean(f0.get_mask()**2)
print(f'{fsky=}')
# fsky = np.sum(apo_mask) / np.size(apo_mask)
# print(f'{fsky=}')

# plt.figure(figsize=(7, 5))
# plt.plot(ls, cl_aa, 'k--')
# plt.plot(ls, pcl_aa/fsky, 'k-', label=r'PCL$_{aa}$/$f_{\rm sky}$')
# plt.plot(ls, cl_ee, 'b--')
# plt.plot(ls, pcl_EE/fsky, 'b-', label=r'PCL$_{EE}$/$f_{\rm sky}$')
# plt.plot(ls, cl_ae, 'r--')
# plt.plot(ls, pcl_aE/fsky, 'r-', label=r'PCL$_{aE}$/$f_{\rm sky}$')
# plt.plot(ls, cl_bb, 'y--')
# plt.plot(ls, pcl_BB/fsky, 'y-', label=r'PCL$_{BB}$/$f_{\rm sky}$')
# plt.plot([], [], '--', c='#AAAAAA', label='Input')
# plt.plot([], [], '-', c='#AAAAAA', label='Simulation')
# plt.yscale('log')
# plt.xlabel(r'$\ell$', fontsize=15)
# plt.ylabel(r'$C_\ell$', fontsize=15)
# plt.legend(fontsize=12, ncol=2)
# plt.show()

delta_ell = 30
b = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell)

# Create a NaMaster workspaces for the different spin combinations
w00 = nmt.NmtWorkspace.from_fields(f0, f0, b)
w02 = nmt.NmtWorkspace.from_fields(f0, f2, b)
w22 = nmt.NmtWorkspace.from_fields(f2, f2, b)

# nsims = 100
# cls_sims_00 = []
# cls_sims_02 = []
# cls_sims_22 = []
# for i in range(nsims):
#     print(f'nsim={i}')
#     a, Q, U = gen_sim()
#     f0 = nmt.NmtField(apo_mask, [a], n_iter=0)
#     f2 = nmt.NmtField(apo_mask, [Q, U], n_iter=0)
#     cls_sims_00.append(w00.decouple_cell(nmt.compute_coupled_cell(f0, f0)))
#     cls_sims_02.append(w02.decouple_cell(nmt.compute_coupled_cell(f0, f2)))
#     cls_sims_22.append(w22.decouple_cell(nmt.compute_coupled_cell(f2, f2)))
# cls_sims_00 = np.array(cls_sims_00)
# cls_sims_02 = np.array(cls_sims_02)
# cls_sims_22 = np.array(cls_sims_22)

# # Compute sim statistics
# cls_mean_00 = np.mean(cls_sims_00, axis=0)
# cls_error_00 = np.std(cls_sims_00, axis=0)/np.sqrt(nsims)
# cls_mean_02 = np.mean(cls_sims_02, axis=0)
# cls_error_02 = np.std(cls_sims_02, axis=0)/np.sqrt(nsims)
# cls_mean_22 = np.mean(cls_sims_22, axis=0)
# cls_error_22 = np.std(cls_sims_22, axis=0)/np.sqrt(nsims)

# # Compute theoretical expectation
# cl_th_00 = w00.decouple_cell(w00.couple_cell([cl_aa]))
# cl_th_02 = w02.decouple_cell(w02.couple_cell([cl_ae, 0*cl_ae]))
# cl_th_22 = w22.decouple_cell(w22.couple_cell([cl_ee, 0*cl_ee, 0*cl_ee, cl_bb]))

# Plot results
leff = b.get_effective_ells()

# fig = plt.figure(figsize=(8, 5))
# ax = fig.add_axes((.1,.3,.8,.6))
# ax.plot(leff, cl_th_00[0], 'k-')
# ax.errorbar(leff, cls_mean_00[0], yerr=cls_error_00[0], fmt='k.', label='$aa$')
# ax.plot(leff, cl_th_02[0], 'r-')
# ax.errorbar(leff, cls_mean_02[0], yerr=cls_error_00[0], fmt='r.', label='$aE$')
# ax.plot(leff, cl_th_22[0], 'b-')
# ax.errorbar(leff, cls_mean_22[0], yerr=cls_error_22[0], fmt='b.', label='$EE$')
# ax.plot(leff, cl_th_22[3], 'y-')
# ax.errorbar(leff, cls_mean_22[3], yerr=cls_error_22[3], fmt='y.', label='$BB$')
# ax.set_yscale('log')
# ax.set_ylabel(r'$C_\ell$', fontsize=15)
# ax.set_xticklabels([])
# ax.legend(ncol=2)
# ax.set_xlim([-0.5, 3*nside])
# ax.axvspan(2*nside, 3*nside, facecolor='#AAAAAA', alpha=0.5)

# ax = fig.add_axes((.1,.1,.8,.2))
# ax.errorbar(leff, (cls_mean_00-cl_th_00)[0]/cls_error_00[0], yerr=np.ones_like(leff), fmt='k.')
# ax.errorbar(leff, (cls_mean_02-cl_th_02)[0]/cls_error_02[0], yerr=np.ones_like(leff), fmt='r.')
# ax.errorbar(leff, (cls_mean_22-cl_th_22)[0]/cls_error_22[0], yerr=np.ones_like(leff), fmt='b.')
# ax.errorbar(leff, (cls_mean_22-cl_th_22)[3]/cls_error_22[3], yerr=np.ones_like(leff), fmt='y.')
# ax.axhline(0, c='k', ls='--')
# ax.set_ylim([-2.9, 2.9])
# ax.set_xlim([-0.5, 3*nside])
# ax.axvspan(2*nside, 3*nside, facecolor='#AAAAAA', alpha=0.5)
# ax.set_ylabel(r'$\Delta C_\ell/\sigma(C_\ell)$', fontsize=16)
# ax.set_xlabel(r'$\ell$', fontsize=16);
# plt.show()


maskbin = apo_mask > 0
f2p = nmt.NmtField(apo_mask, [mpQ*maskbin, mpU*maskbin], purify_b=True)

# Now compute the MCM
w22p = nmt.NmtWorkspace.from_fields(f2p, f2p, b)

nsims = 200
clbb_sims = []
clbb_sims_pure = []
for i in range(nsims):
    print(f'sim={i}')
    _, Q, U = gen_sim()
    Q *= maskbin
    U *= maskbin
    f = nmt.NmtField(apo_mask, [Q, U])
    fp = nmt.NmtField(apo_mask, [Q, U], purify_b=True)
    clbb_sims.append(w22.decouple_cell(nmt.compute_coupled_cell(f, f))[3])
    clbb_sims_pure.append(w22p.decouple_cell(nmt.compute_coupled_cell(fp, fp))[3])
clbb_sims = np.array(clbb_sims)
clbb_sims_pure = np.array(clbb_sims_pure)

cl_mean = np.mean(clbb_sims, axis=0)
cl_error = np.std(clbb_sims, axis=0)/np.sqrt(nsims)
cl_mean_pure = np.mean(clbb_sims_pure, axis=0)
cl_error_pure = np.std(clbb_sims_pure, axis=0)/np.sqrt(nsims)

cl_th = w22.decouple_cell(w22.couple_cell([cl_ee, 0*cl_ee, 0*cl_ee, cl_bb]))[3]
cl_th_pure = w22p.decouple_cell(w22p.couple_cell([cl_ee, 0*cl_ee, 0*cl_ee, cl_bb]))[3]

# First let's check that the power spectra are unbiased.
# Notice that, after accounting for residual mode coupling,
# the theoretical prediction for C_ell^BB depends on whether
# purification has been carried out.
fig = plt.figure(figsize=(8, 5))
ax = fig.add_axes((.1,.3,.8,.6))
ax.plot(leff, cl_th, 'r-', label='Input (no purif)')
ax.plot(leff, cl_th_pure, 'b-', label='Input (w. purif)')
ax.errorbar(leff, cl_mean, yerr=cl_error, fmt='r.', label='Sims (no purif.)')
ax.errorbar(leff, cl_mean_pure, yerr=cl_error_pure, fmt='b.', label='Sims (w. purif.)')
ax.set_yscale('log')
ax.set_ylabel(r'$C_\ell$', fontsize=15)
ax.set_xticklabels([])
ax.legend(ncol=2)
ax.set_xlim([-0.5, 3*nside])
ax.axvspan(2*nside, 3*nside, facecolor='#AAAAAA', alpha=0.5)

ax = fig.add_axes((.1,.1,.8,.2))
ax.errorbar(leff, (cl_mean-cl_th)/cl_error, yerr=np.ones_like(leff), fmt='r.')
ax.errorbar(leff, (cl_mean_pure-cl_th_pure)/cl_error_pure, yerr=np.ones_like(leff), fmt='b.')
ax.axhline(0, c='k', ls='--')
ax.set_ylim([-2.9, 2.9])
ax.set_xlim([-0.5, 3*nside])
ax.axvspan(2*nside, 3*nside, facecolor='#AAAAAA', alpha=0.5)
ax.set_ylabel(r'$\Delta C_\ell/\sigma(C_\ell)$', fontsize=16)
ax.set_xlabel(r'$\ell$', fontsize=16);

# And now let's look at the error bar
plt.figure(figsize=(7, 5))
plt.plot(leff, cl_error, 'r-', label='No purification')
plt.plot(leff, cl_error_pure, 'b-', label='W. purification')
plt.yscale('log')
plt.xlabel(r'$\ell$', fontsize=15)
plt.ylabel(r'$\sigma(C^{BB}_\ell)$', fontsize=15)
plt.legend(fontsize=12, ncol=2)

plt.show()
