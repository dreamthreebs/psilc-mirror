import healpy as hp
import numpy as np
import pymaster as nmt
import matplotlib.pyplot as plt

nside = 128
lmax = 3 * nside - 1
np.random.seed(42)

# ==== Step 1: Generate two uncorrelated Gaussian polarization maps ====

# Simulate spectra for A and B components
cl_A = np.zeros((4, lmax + 1))  # EE, BB
cl_B = np.zeros((4, lmax + 1))
cl_A[0, 2:] = 1e-5 / (np.arange(2, lmax + 1) * (np.arange(2, lmax + 1) + 1))  # EE
cl_A[1, 2:] = 5e-6 / (np.arange(2, lmax + 1) * (np.arange(2, lmax + 1) + 1))  # BB
cl_B[0, 2:] = 2e-5 / (np.arange(2, lmax + 1) * (np.arange(2, lmax + 1) + 1))
cl_B[1, 2:] = 1e-5 / (np.arange(2, lmax + 1) * (np.arange(2, lmax + 1) + 1))

map_A = hp.synfast(cl_A, nside=nside, new=True, pol=True)  # [I, Q, U]
map_B = hp.synfast(cl_B, nside=nside, new=True, pol=True)
map_M = map_A + map_B

# ==== Step 2: Define mask and beam ====

mask = hp.ud_grade(np.ones(hp.nside2npix(32)), nside_out=nside)
mask = hp.smoothing(mask, fwhm=np.radians(1.0))  # soft apodization
mask[mask < 0.9] = 0  # threshold to binary
mask[mask >= 0.9] = 1

beam = hp.gauss_beam(np.radians(30 / 60), lmax=lmax)  # 30 arcmin beam

# ==== Step 3: Define binning ====
binning = nmt.NmtBin.from_nside_linear(nside=nside, nlb=20)

# ==== Step 4: Run verification ====
from typing import Dict

def verify_power_spectrum_additivity(
    mA_q, mA_u,
    mB_q, mB_u,
    beam, mask, binning,
    purify_b=True,
    masked_on_input=True,
    lmax=3*512-1,
    plot=True
) -> Dict[str, np.ndarray]:
    def build_field(q, u):
        return nmt.NmtField(mask, [q, u], beam=beam,
                            purify_b=purify_b,
                            masked_on_input=masked_on_input,
                            lmax=lmax, lmax_mask=lmax)

    # Construct maps
    mM_q = mA_q + mB_q
    mM_u = mA_u + mB_u

    # Fields
    fA = build_field(mA_q, mA_u)
    fB = build_field(mB_q, mB_u)
    fM = build_field(mM_q, mM_u)

    # Workspace
    wsp = nmt.NmtWorkspace()
    wsp.compute_coupling_matrix(fA, fA, binning)

    # Coupled and Decoupled spectra
    cMM = wsp.decouple_cell(nmt.compute_coupled_cell(fM, fM))[3]
    cAA = wsp.decouple_cell(nmt.compute_coupled_cell(fA, fA))[3]
    cBB = wsp.decouple_cell(nmt.compute_coupled_cell(fB, fB))[3]
    cAB = wsp.decouple_cell(nmt.compute_coupled_cell(fA, fB))[3]

    prediction = cAA + cBB + 2 * cAB
    prediction_uncorr = cAA + cBB
    ells = binning.get_effective_ells()

    if plot:
        plt.figure()
        plt.plot(ells, cMM, label=r"$C_\ell^{MM}$")
        plt.plot(ells, prediction, label=r"$C_\ell^{AA} + C_\ell^{BB} + 2 C_\ell^{AB}$", linestyle='--')
        plt.plot(ells, prediction_uncorr, label=r"$C_\ell^{AA} + C_\ell^{BB}$", linestyle=':')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$C_\ell^{BB}$')
        plt.legend()
        plt.title("Power Spectrum Linearity Check")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {
        "ells": ells,
        "C_MM": cMM,
        "C_AA": cAA,
        "C_BB": cBB,
        "C_AB": cAB,
    }

# === Run ===
result = verify_power_spectrum_additivity(
    mA_q=map_A[1], mA_u=map_A[2],
    mB_q=map_B[1], mB_u=map_B[2],
    beam=beam,
    mask=mask,
    binning=binning,
    lmax=lmax,
)

