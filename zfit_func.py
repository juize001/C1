import zfit
import numpy as np
from analysis_func import get_val_err

def fit_asymmetry_for_dataset(df):
    # Split by charge
    positive_data = df[df['B_assumed_particle_type'] > 0]
    negative_data = df[df['B_assumed_particle_type'] < 0]

    # Define observable and datasets
    obs = zfit.Space('mass', limits=(5200, 5400))
    data_plus  = zfit.Data.from_numpy(obs=obs, array=positive_data['B_invariant_mass'].values)
    data_minus = zfit.Data.from_numpy(obs=obs, array=negative_data['B_invariant_mass'].values)

    # Parameters (shared shapes)
    mean  = zfit.Parameter('mean', 5283.8, 5200., 5360.)
    sigma = zfit.Parameter('sigma', 18.6, 1e-3, 200.)
    expo  = zfit.Parameter('expo', -0.003, -1.0, 1.0)

    # Yields
    Nsig_plus  = zfit.Parameter("Nsig_plus",  700., 0., 1e6)
    Nbkg_plus  = zfit.Parameter("Nbkg_plus", 3000., 0., 1e6)
    Nsig_minus = zfit.Parameter("Nsig_minus",  740., 0., 1e6)
    Nbkg_minus = zfit.Parameter("Nbkg_minus", 3200., 0., 1e6)

    # PDFs
    gauss = zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma)
    bkg   = zfit.pdf.Exponential(obs=obs, lambda_=expo)

    sig_plus  = gauss.create_extended(Nsig_plus)
    bkg_plus  = bkg.create_extended(Nbkg_plus)
    sig_minus = gauss.create_extended(Nsig_minus)
    bkg_minus = bkg.create_extended(Nbkg_minus)

    model_plus  = zfit.pdf.SumPDF([sig_plus, bkg_plus])
    model_minus = zfit.pdf.SumPDF([sig_minus, bkg_minus])

    # Loss
    nll = zfit.loss.ExtendedUnbinnedNLL(model=[model_plus, model_minus],
                                        data=[data_plus, data_minus])

    # Minimize
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(nll)

    # HESSE errors
    try:
        result.hesse()
    except Exception:
        pass
    print(result)
    print(f"Function minimum: {result.fmin}")
    print(f"Converged: {result.converged}")
    print(f"Valid: {result.valid}")

    val_Np, err_Np = get_val_err(result, Nsig_plus)
    val_Nm, err_Nm = get_val_err(result, Nsig_minus)

    # Asymmetry and uncertainty
    A_raw = (val_Nm - val_Np) / (val_Nm + val_Np)
    A_raw_err = 2 / (val_Nm + val_Np)**2 * np.sqrt((val_Np * err_Nm)**2 + (val_Nm * err_Np)**2)

    return A_raw, A_raw_err, val_Np, val_Nm