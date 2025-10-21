import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
sns.set_style("darkgrid")

########## TO DO LIST - SELECTIONS - https://arxiv.org/pdf/1408.0978
# 1. (done) Apply cut to D meson resonance range (1850-1880MeV)
# 2. (done) Veto Charmonium backgrounds from hadron misidentification to muon
# 3. (done) Veto some particle misidentifications e.g. kaons and pions
# 4. (done) Veto kaon and muon swapped mass of jpsik decay
# 5. Research standard model predictions for CP asymmetry based on dimuon mass squared
# 6. Test the fit bias e.g. generate some data based on fit PDF and calculate the goodness of fit for each set
# 7. Quantify accuracy and any biases of the ML model, include statistics and graphs

# detector resolution for invariant mass - https://lhcb.web.cern.ch/speakersbureau/html/PerformanceNumbers.html
# paper on LHCb setup and detector performance - https://iopscience.iop.org/article/10.1088/1748-0221/3/08/S08005/pdf

from analysis_func import *

# data_2011 = pd.read_pickle('LHCb/dataset_2011.pkl')
# samesign_2011 = pd.read_pickle('LHCb/samesign_2011.pkl')
samesign_2012 = pd.read_pickle('LHCb/samesign_2012.pkl')
data_2012u = pd.read_pickle('LHCb/dataset_2012_MagnetUp.pkl')
data_2012d = pd.read_pickle('LHCb/dataset_2012_MagnetDown.pkl')
data_2012 = pd.concat([data_2012u, data_2012d])

data_2012.columns = [re.sub(r'[^A-Za-z0-9_]', '_', c) for c in data_2012.columns]
samesign_2012.columns = [re.sub(r'[^A-Za-z0-9_]', '_', c) for c in samesign_2012.columns]
signal_data = data_2012[(abs(data_2012['B_invariant_mass'] - 5280) < 150) & (abs(data_2012['dimuon_system_invariant_mass'] - 3097) < 50)]
non_resonance_data = data_2012[(data_2012['dimuon_system_invariant_mass'] < 2828) | (data_2012['dimuon_system_invariant_mass'] > 3873) | ((data_2012['dimuon_system_invariant_mass'] > 3317) & (data_2012['dimuon_system_invariant_mass'] < 3536))]

# signal_data = data_2011[abs(data_2011['B invariant mass'] - 5280) < 200]
# non_resonance_data_binned, signal_data, _ = split_into_q2_bins(data_2012.copy())
# non_resonance_data = pd.concat(non_resonance_data_binned.values(), ignore_index=True)
background_data = samesign_2012

# forbidden_vars = ['B_invariant_mass', 'dimuon_system_invariant_mass', 'Event_ID', 'Magnet_polarity']
training_labels = ['Kaon_impact_parameter_chi2_wrt_primary_vertex', 'B_decay_vertex_fit_chi2', 'Kaon_PID_NN_score_for_muon_hypothesis', 'dimuon_system_flight_distance_wrt_B_decay_vertex', 'Isolation__B_vertex_delta_chi2_adding_two_extra_tracks__best_fits_', 'B_assumed_particle_type', 'Opposite_sign_muon_PID_NN_score_for_muon_hypothesis', 'Same_sign_muon_PID_NN_score_for_muon_hypothesis', 'Kaon_PID_NN_score_for_kaon_hypothesis', 'B_decay_vertex_x_position', 'B_cos_angle__between_line_of_flight_and_momentum', 'Isolation__B_mass_if_one_extra_track__best_fit__is_added', 'B_4_momentum_x_component', 'Isolation__B_vertex_delta_chi2_adding_one_extra_track__best_fit_', 'dimuon_system_impact_parameter_chi2_wrt_primary_vertex', 'B_impact_parameter_wrt_primary_vertex', 'dimuon_system_flight_distance_chi2_wrt_primary_vertex', 'B_decay_vertex_y_position', 'dimuon_system_cos_angle__between_line_of_flight_from_primary_vertex_and_momentum', 'B_magnitude_of_momentum_transverse_to_beam']
training_labels = '|'.join(training_labels)
# resonance_data = data[(abs(data['B invariant mass'] - 5280) < 150) & (abs(data['dimuon-system invariant mass'] - 3097) < 50)]

# positive_data = signal_data[signal_data['B assumed particle type'] > 0]
# negative_data = signal_data[signal_data['B assumed particle type'] < 0]
# jpsik_acp, jpsik_acp_err = acp_calc(positive_data, negative_data)
# print(f'CP Asymmetry (raw) for JPsiK decay is {jpsik_acp} +- {jpsik_acp_err}')
# exit()

# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import HistGradientBoostingClassifier
# from sklearn.inspection import permutation_importance
import lightgbm as lgb

from training_func import *
#lgbm, y_test, y_pred = train_model(signal_data, background_data, test_size=0.001)
#joblib.dump(lgbm, 'Models/model_ss2012_20_full.pkl')
#lgbm = joblib.load('Models/model_ss2012.pkl')
lgbm = joblib.load('Models/model_ss2012_20_full.pkl')

from zfit_func_pulls import fit_asymmetry_cb

optimal_t, results_df = find_optimal_threshold(
    data=non_resonance_data,
    model=lgbm,
    training_labels=training_labels,
    fit_func=fit_asymmetry_cb)
exit()



resonance_data = apply_model(signal_data.copy(), lgbm, threshold=0.95)
positive_data = resonance_data[resonance_data['B_assumed_particle_type'] > 0]
negative_data = resonance_data[resonance_data['B_assumed_particle_type'] < 0]
jpsik_acp, jpsik_acp_err = acp_calc(positive_data, negative_data)
print(f'CP Asymmetry (raw) for JPsiK decay is {jpsik_acp} +- {jpsik_acp_err}')



high_conf_signal = apply_model(non_resonance_data.copy(), lgbm)
# plt.hist(non_resonance_data['B_invariant_mass'], bins=1000)
# plt.hist(high_conf_signal['B_invariant_mass'], bins=1000)
# # plt.savefig(f'/Users/zifei/Desktop/samesign_trained_95.png', dpi=300)
# plt.show()
# high_conf_signal.to_pickle("/Users/zifei/Desktop/C1/ProcessedData/first_selection_data3.pkl")


# --- perform fit bias analysis and calculate pulls --- #
if 1 == 2:
    from zfit_func_pulls import *
    import tensorflow as tf
    binned_data, _, bin_bounds = split_into_q2_bins(high_conf_signal.copy())
    obs = zfit.Space('mass', limits=(5200, 5600))

    all_pulls = {}

    for bin_idx, dat in binned_data.items():
        if dat.empty:
            continue
        
        print(f"\n=== Q² bin {bin_idx} ===")
        A_raw, A_raw_err, Np, Nm, fit_params = fit_asymmetry_cb(dat)
        # Build zfit model from fitted parameters
        model_plus, model_minus = build_model(fit_params, obs)
        # Run toy study
        pulls, failed = run_toy_study(model_plus, model_minus, fit_params, ntoys=100)
        print(f"Toys completed: {len(pulls)}, failed: {failed}")

        all_pulls[bin_idx] = {
        "pulls": pulls,
        "failed": failed,
        "fit_params": fit_params
        }

        # --- SAVE AFTER EACH BIN ---
        joblib.dump(all_pulls, "pulls_per_q2_bin.pkl")
        print("Progress saved.")

        # --- RESET GRAPH after each bin ---
        tf.compat.v1.reset_default_graph()
        print("Graph reset.\n")

        # Plot
        mean, std = plot_pulls(pulls, title=f"Q² bin {bin_idx}")
        print(f"Pull mean={mean:.3f}, std={std:.3f}")

    exit()


# from zfit_func import *
# from zfit_func_pulls import *
# high_conf_signal = high_conf_signal[(high_conf_signal['B_invariant_mass'] > 5000) & (high_conf_signal['B_invariant_mass'] < 5500)]
# sss = high_conf_signal[(high_conf_signal['B_invariant_mass'] < 5700) & (high_conf_signal['B_invariant_mass'] > 5150)].copy()
high_conf_signal_with_vetoes = post_selection_vetoes(high_conf_signal.copy(), diagnostics=True)
exit()

A_raw_tot, A_raw_err_tot, *_ = fit_asymmetry_cb(high_conf_signal_with_vetoes.copy())
A_raw_tot = A_raw_tot - jpsik_acp
A_raw_err_tot = np.sqrt(A_raw_err_tot ** 2 + jpsik_acp_err ** 2)
print(f'Corrected CP Asymmetry (raw) for Kmumu decay is {A_raw_tot} +- {A_raw_err_tot}')

binned_selected_data, _, bin_bounds = split_into_q2_bins(high_conf_signal_with_vetoes.copy())
results = []
for bin_idx, dat in binned_selected_data.items():  # .items() gives both key and value
    dat = kmu_mass_filter(dat)
    if dat.empty:
        continue
    try:
        A_raw, A_raw_err, val_Np, val_Nm, _ = fit_asymmetry_cb(dat)
    except Exception as e:
        print(f"Bin {bin_idx} failed: {e}")
        continue
    A_raw = A_raw - jpsik_acp
    A_raw_err = np.sqrt(A_raw_err ** 2 + jpsik_acp_err ** 2)

    q2_low = bin_bounds[bin_idx]
    q2_high = bin_bounds[bin_idx + 1]
    q2_center = 0.5 * (q2_low + q2_high)

    results.append({
        "q2_bin": bin_idx,
        "q2_low": q2_low,
        "q2_high": q2_high,
        "q2_center": q2_center,
        "A_raw": A_raw,
        "A_raw_err": A_raw_err,
        "Nsig_plus": val_Np,
        "Nsig_minus": val_Nm,
        "n_events": len(dat)
    })
binned_data_all = pd.DataFrame(results).sort_values("q2_center").reset_index(drop=True)
plt.errorbar(binned_data_all["q2_center"], binned_data_all["A_raw"],
             yerr=binned_data_all["A_raw_err"], fmt='o', capsize=4)
for _, row in binned_data_all.iterrows():
    plt.text(
        row["q2_center"], row["A_raw"] + 0.005,
        f"N⁺={int(row['Nsig_plus'])}\nN⁻={int(row['Nsig_minus'])}",
        ha='center', va='bottom', fontsize=7, color='dimgray'
    )
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
plt.axhline(y=A_raw_tot, color='black', linestyle='--', linewidth=1)
plt.fill_between(
    binned_data_all["q2_center"],
    A_raw_tot - A_raw_err_tot,  # lower edge of band
    A_raw_tot + A_raw_err_tot,  # upper edge of band
    color='gray',
    alpha=0.2,
    label=r'$\pm 0.02$ uncertainty zone'
)
for xval in [8, 11, 12.5, 15]:
    plt.axvline(x=xval, color='red', linestyle='-', linewidth=1)
plt.xlabel("Dimuon Mass Squared")
plt.ylabel("Raw Asymmetry  $A_{raw}$")
plt.title("Corrected Asymmetry vs Dimuon Mass Squared")
plt.grid()
plt.tight_layout()
plt.show()
exit()


