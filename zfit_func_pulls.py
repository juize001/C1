import zfit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from analysis_func import get_val_err

def fit_asymmetry_cb(df):
    # Split by charge
    positive_data = df[df['B_assumed_particle_type'] > 0]
    negative_data = df[df['B_assumed_particle_type'] < 0]

    # Define observable and datasets
    obs = zfit.Space('mass', limits=(5200, 5600))
    data_plus  = zfit.Data.from_numpy(obs=obs, array=positive_data['B_invariant_mass'].values)
    data_minus = zfit.Data.from_numpy(obs=obs, array=negative_data['B_invariant_mass'].values)

    # Parameters (shared shapes)
    mean  = zfit.Parameter('mean', 5283.8, 5200., 5360.)
    sigma = zfit.Parameter('sigma', 18.6, 1e-3, 200.)
    alpha = zfit.Parameter('alpha', 1.5, 0.1, 10.0)
    n = zfit.Parameter('n', 3.0, 1.1, 50.0)
    expo  = zfit.Parameter('expo', -0.003, -1.0, 1.0)


    # Yields
    Nsig_plus  = zfit.Parameter("Nsig_plus",  700., 0., 1e6)
    Nbkg_plus  = zfit.Parameter("Nbkg_plus", 3000., 0., 1e6)
    Nsig_minus = zfit.Parameter("Nsig_minus",  740., 0., 1e6)
    Nbkg_minus = zfit.Parameter("Nbkg_minus", 3200., 0., 1e6)

    # PDFs
    #gauss = zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma)
    cb = zfit.pdf.CrystalBall(obs=obs, mu=mean, sigma=sigma, alpha=alpha, n=n)
    bkg   = zfit.pdf.Exponential(obs=obs, lambda_=expo)

    sig_plus  = cb.create_extended(Nsig_plus)
    bkg_plus  = bkg.create_extended(Nbkg_plus)
    sig_minus = cb.create_extended(Nsig_minus)
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
        #print(result)
    except Exception:
        print('ZFIT FIT CONVERGENCE HAS FAILED, CHECK THE DATA OR FIT PARAMETERS!!!!!')
        pass
    #print(f"Function minimum: {result.fmin}")
    #print(f"Converged: {result.converged}")
    #print(f"Valid: {result.valid}")
    

    val_Np, err_Np = get_val_err(result, Nsig_plus)
    val_Nm, err_Nm = get_val_err(result, Nsig_minus)

    # Asymmetry and uncertainty
    A_raw = (val_Nm - val_Np) / (val_Nm + val_Np)
    A_raw_err = 2 / (val_Nm + val_Np)**2 * np.sqrt((val_Np * err_Nm)**2 + (val_Nm * err_Np)**2)

    return A_raw, A_raw_err, val_Np, val_Nm, {
        "mean": mean.value(),
        "sigma": sigma.value(),
        "alpha": alpha.value(),
        "n": n.value(),
        "expo": expo.value(),
        "Nsig_plus": Nsig_plus.value(),
        "Nsig_minus": Nsig_minus.value(),
        "Nbkg_plus": Nbkg_plus.value(),
        "Nbkg_minus": Nbkg_minus.value(),
        }


def build_model(fit_params, obs):

    mean  = zfit.Parameter('mean_gen', fit_params['mean'], floating=False)
    sigma = zfit.Parameter('sigma_gen', fit_params['sigma'], floating=False)
    alpha = zfit.Parameter('alpha_gen', fit_params['alpha'], floating=False)
    n     = zfit.Parameter('n_gen', fit_params['n'], floating=False)
    expo  = zfit.Parameter('expo_gen', fit_params['expo'], floating=False)

    # PDFs
    cb   = zfit.pdf.CrystalBall(obs=obs, mu=mean, sigma=sigma, alpha=alpha, n=n)
    bkg  = zfit.pdf.Exponential(obs=obs, lambda_=expo)

    # Yields
    Nsig_plus  = int(fit_params['Nsig_plus'])
    Nbkg_plus  = int(fit_params['Nbkg_plus'])
    Nsig_minus = int(fit_params['Nsig_minus'])
    Nbkg_minus = int(fit_params['Nbkg_minus'])

    sig_plus  = cb.create_extended(Nsig_plus)
    bkg_plus  = bkg.create_extended(Nbkg_plus)
    sig_minus = cb.create_extended(Nsig_minus)
    bkg_minus = bkg.create_extended(Nbkg_minus)

    model_plus  = zfit.pdf.SumPDF([sig_plus, bkg_plus])
    model_minus = zfit.pdf.SumPDF([sig_minus, bkg_minus])

    return model_plus, model_minus


def generate_toys(model_plus, model_minus, ntoys=100):

    toy_datasets = []

    for i in range(ntoys):
        toy_data_plus = model_plus.sample()
        toy_data_minus = model_minus.sample()
        toy_datasets.append((toy_data_plus, toy_data_minus))

    return toy_datasets


def toy_to_df(toy_plus, toy_minus):

    plus_array = toy_plus.numpy().flatten()
    minus_array = toy_minus.numpy().flatten()

    df_plus = pd.DataFrame({
        'B_invariant_mass': plus_array,
        'B_assumed_particle_type': np.ones_like(plus_array)
    })
    df_minus = pd.DataFrame({
        'B_invariant_mass': minus_array,
        'B_assumed_particle_type': -1*np.ones_like(minus_array)
    })

    df_toy = pd.concat([df_plus, df_minus], ignore_index=True)

    return df_toy


def fit_toy(toy_plus, toy_minus):

    df_toy = toy_to_df(toy_plus, toy_minus)
    A_fit, A_fit_err, Np_fit, Nm_fit, _= fit_asymmetry_cb(df_toy)

    return A_fit, A_fit_err, Np_fit, Nm_fit


def run_toy_study(model_plus, model_minus, fit_params, ntoys=100):

    A_true = (fit_params['Nsig_minus'] - fit_params['Nsig_plus']) / \
             (fit_params['Nsig_minus'] + fit_params['Nsig_plus'])

    pulls = []
    failed = 0

    for i in range(ntoys):
        # print(f'Currently on toydata number {i}')
        toy_plus = model_plus.sample()
        toy_minus = model_minus.sample()

        try:
            A_fit, A_fit_err, *_ = fit_toy(toy_plus, toy_minus)
            pull = (A_fit - A_true) / A_fit_err
            pulls.append(pull)
        except Exception as e:
            print(f"Toy {i} failed: {e}")
            failed += 1

    return np.array(pulls), failed


def plot_pulls(pulls, title="Pull Distribution"):
    plt.hist(pulls, bins=30, density=True, alpha=0.7, color="steelblue")
    mean, std = np.mean(pulls), np.std(pulls)
    plt.axvline(0, color='red', linestyle='--')
    plt.title(f"{title}\nmean={mean:.3f}, std={std:.3f}")
    plt.xlabel("(Fitted - True) / σ_fit")
    plt.ylabel("Density")
    # plt.savefig(f"plots/pulls_q2bin{bin_idx}.png")
    plt.show()
    return mean, std



