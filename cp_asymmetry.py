import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zfit

from analysis_func import kmu_mass_filter

ss_data = pd.read_pickle("/Users/zifei/Desktop/C1/ProcessedData/first_selection_data3.pkl")
ss_data = kmu_mass_filter(ss_data)
positive_data = ss_data[ss_data['B_assumed_particle_type'] > 0]
negative_data = ss_data[ss_data['B_assumed_particle_type'] < 0]

obs = zfit.Space('mass', limits=(5200, 5400))   # adapt mass window
data_plus  = zfit.Data.from_numpy(obs=obs, array=positive_data['B_invariant_mass'].values)
data_minus = zfit.Data.from_numpy(obs=obs, array=negative_data['B_invariant_mass'].values)

# --- 1. Parameters (yields are floating, shape parameters shared)
N_plus  = zfit.Parameter('N_plus',  max(1, len(positive_data)//2),  0., 1e6)
N_minus = zfit.Parameter('N_minus', max(1, len(negative_data)//2), 0., 1e6)

# Shared shape parameters (example)
mean  = zfit.Parameter('mean', 5279.5, 5200., 5360.)
try:
    mean.set_constant(True)   # common method
except Exception:
    # fallback: set floating attribute
    mean.floating = False
sigma = zfit.Parameter('sigma', 18.6, 1e-3, 200.)
expo  = zfit.Parameter('expo', -0.003, -1.0, 1.0)

# --- 2. PDFs (substitute your preferred signal PDF)
gauss = zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma)
bkg   = zfit.pdf.Exponential(obs=obs, lambda_=expo)   # zfit uses lambda_ keyword

# Extended combined (signal + bkg) for each charge. We create separate models so we can
# let the yields vary independently while sharing shapes.
# --- Extended PDFs (modern zfit pattern)
Nsig_plus  = zfit.Parameter("Nsig_plus",  700., 0., 1e6)
Nbkg_plus  = zfit.Parameter("Nbkg_plus", 3000., 0., 1e6)
Nsig_minus = zfit.Parameter("Nsig_minus",  740., 0., 1e6)
Nbkg_minus = zfit.Parameter("Nbkg_minus", 3200., 0., 1e6)

sig_plus  = gauss.create_extended(Nsig_plus)
bkg_plus  = bkg.create_extended(Nbkg_plus)
sig_minus = gauss.create_extended(Nsig_minus)
bkg_minus = bkg.create_extended(Nbkg_minus)

model_plus  = zfit.pdf.SumPDF([sig_plus,  bkg_plus])
model_minus = zfit.pdf.SumPDF([sig_minus, bkg_minus])

# Alternatively use zfit.pdf.ExtendedPDF to make them explicit extended PDFs:
# If your zfit version provides ExtendedPDF wrapper, use it; otherwise ExtendedUnbinnedNLL will handle counts.

# --- 3. Build simultaneous loss: pass lists of models/data to ExtendedUnbinnedNLL
nll = zfit.loss.ExtendedUnbinnedNLL(model=[model_plus, model_minus],
                                    data=[data_plus, data_minus])

# --- 4. Minimize
minimizer = zfit.minimize.Minuit()   # or zfit.minimize.Scipy or Minuit with options
# Define the minimizer
# Run the minimization
result = minimizer.minimize(nll)

# (Optional) run error estimation
result.hesse()

# Note: use zfit.minimize.minimize(...) — wrapper returns a FitResult

# --- 5. Compute/obtain covariance with HESSE (recommended). Try different methods if needed.
# FitResult has methods to run hesse and get covariance. API differs across versions:
try:
    # modern API: run hesse and then get covariance for the two parameters
    result.hesse()   # computes/updates result internal covariance (may call Minuit.hesse)
except Exception as e:
    # fallback: try the numerical hesse implementation
    try:
        result.hesse(method='hesse_np')
    except Exception:
        print("Hesse failed; you can try result.hesse(method='hesse_np') or use Minuit/Minos alternatives", e)

# get parameter values and build covariance matrix for [N_plus, N_minus]
# FitResult.params is typically a dict {Parameter: value}
params = result.params  # dict-like
def get_val_err(result, param):
    info = result.params[param]
    val = info["value"]
    # Try several possible locations for the uncertainty
    err = (
        info.get("error")
        or (info.get("hesse") or {}).get("error")
        or (info.get("minuit_hesse") or {}).get("error")
        or (info.get("minuit_minos") or {}).get("upper")  # fallback
        or 0.0
    )
    return val, err

val_Np, err_Np = get_val_err(result, Nsig_plus)
val_Nm, err_Nm = get_val_err(result, Nsig_minus)

A_raw = (val_Np - val_Nm) / (val_Nm + val_Np)
A_raw_err = 2 / (val_Nm + val_Np)**2 * np.sqrt((val_Np * err_Nm)**2 + (val_Nm * err_Np)**2)

print(f"A_raw = {A_raw:.4f} ± {A_raw_err:.4f}")
print(result.params[Nsig_plus])
print(result.params[Nsig_minus])
