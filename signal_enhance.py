import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
# sns.set_style("darkgrid")

########## TO DO LIST - SELECTIONS - https://arxiv.org/pdf/1408.0978 - https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.111.151801
# 1. Deal with unknown peak/resonance in Kmu mass plotted under dimuon mass hypothesis
# 2. Get optimised model parameters and threshold
# 3. Calculate ACP value using formula given in papers (weighted ACP)
# 4. Get uncertainties from different fit functions

# detector resolution for invariant mass - https://lhcb.web.cern.ch/speakersbureau/html/PerformanceNumbers.html
# paper on LHCb setup and detector performance - https://iopscience.iop.org/article/10.1088/1748-0221/3/08/S08005/pdf

def calc_acp_q2_bins(data, visual=False):
    binned_selected_data, _, bin_bounds = split_into_q2_bins(data.copy())
    results = []
    all_pulls = joblib.load("binned_pulls2.pkl")
    for bin_idx, dat in binned_selected_data.items():  # .items() gives both key and value
        dat = post_selection_vetoes(dat)

        if dat.empty:
            continue
        try:
            A_raw, A_raw_err, val_Np, val_Nm, *_ = fit_asymmetry_cb(dat)
            print(f'Bin {bin_idx} has been fitted.')
        except Exception as e:
            print(f"Bin {bin_idx} failed: {e}")
            continue

        pulls = np.array(all_pulls[bin_idx]['pulls'])
        pulls = pulls[np.isfinite(pulls)]
        N_pulls = len(pulls)
        pull_mean = float(np.mean(pulls))
        pull_std  = float(np.std(pulls, ddof=1))

        # Absolute bias on A_raw 
        bias = pull_mean * A_raw_err
        # Inflate statistical error if pull width != 1
        stat_err = A_raw_err * pull_std

        fit_syst_err = A_raw_err * pull_std / np.sqrt(N_pulls)

        # total systematic errors
        fit_syst_err = np.sqrt(bias**2 + fit_syst_err**2)
        sys_err = np.sqrt(jpsik_acp_err**2 + fit_syst_err**2)

        Acp = A_raw - jpsik_acp
        A_err = np.sqrt(stat_err ** 2 + jpsik_acp_err ** 2 + fit_syst_err ** 2)

        q2_low = bin_bounds[bin_idx]
        q2_high = bin_bounds[bin_idx + 1]
        q2_center = 0.5 * (q2_low + q2_high)
        q2_width = 0.5 * abs(q2_high - q2_low)

        results.append({
            "q2_bin": bin_idx,
            "q2_low": q2_low,
            "q2_high": q2_high,
            "q2_center": q2_center,
            "q2_width": q2_width,
            "Acp": Acp,
            "A_err": A_err,
            "stat_err": stat_err,
            "sys_err": sys_err,
            "A_raw_err": A_raw_err,
            "jpsik_err": jpsik_acp_err,
            "fit_syst_err": fit_syst_err,
            "Nsig_plus": val_Np,
            "Nsig_minus": val_Nm,
            "n_events": len(dat)
        })
    binned_data_all = pd.DataFrame(results).sort_values("q2_center").reset_index(drop=True)

    if visual:
        for xval in [8, 11, 12.5, 15]:
            plt.axvline(x=xval, color='red', linestyle='-', linewidth=1)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
        plt.axhline(y=A_raw_tot, color='black', linestyle='--', linewidth=1)
        plt.errorbar(binned_data_all["q2_center"], binned_data_all["Acp"],
                    yerr=binned_data_all["A_err"], xerr=binned_data_all["q2_width"], fmt='o', capsize=4)
        # for _, row in binned_data_all.iterrows():
        #     plt.text(
        #         row["q2_center"], row["A_raw"] + 0.005,
        #         f"N⁺={int(row['Nsig_plus'])}\nN⁻={int(row['Nsig_minus'])}",
        #         ha='center', va='bottom', fontsize=7, color='dimgray'
        #     )
        
        plt.fill_between(
            binned_data_all["q2_center"],
            A_raw_tot - A_raw_err_tot,  # lower edge of band
            A_raw_tot + A_raw_err_tot,  # upper edge of band
            color='gray',
            alpha=0.2,
            label=r'$\pm 0.02$ uncertainty zone'
        )
        plt.xlabel("Dimuon Mass Squared (GeV$^2$/$c^4$)")
        plt.ylabel("CP Asymmetry")
        # plt.title("Corrected Asymmetry vs Dimuon Mass Squared")
        plt.grid()
        plt.tight_layout()
        plt.show()
    return binned_data_all

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
# lgbm, y_test, y_pred = train_model(signal_data, background_data, test_size=0.999, class_output=True)
#joblib.dump(lgbm, 'Models/model_ss2012_20_full.pkl')
lgbm = joblib.load('Models/model_ss2012_20_full.pkl')


from zfit_func_pulls import fit_asymmetry_cb

if 1==2:
    optimal_t, results_df = find_threshold(
        data=non_resonance_data,
        model=lgbm,
        training_labels=training_labels,
        fit_function=fit_asymmetry_cb)
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
    import gc
    import tensorflow as tf
    # binned_data, _, bin_bounds = split_into_q2_bins(high_conf_signal.copy())
    obs = zfit.Space('mass', limits=(5150, 5600))

    binned_data = {0: high_conf_signal} # test entire dataset
    all_pulls = {}

    for bin_idx, dat in binned_data.items():
        if dat.empty:
            continue
        
        print(f"\n=== Q² bin {bin_idx} ===")
        A_raw, A_raw_err, Np, Nm, fit_params = fit_asymmetry_cb(dat)
        # Build zfit model from fitted parameters
        model_plus, model_minus = build_model(fit_params, obs)
        # Run toy study
        pulls, failed = run_toy_study(model_plus, model_minus, fit_params, ntoys=200)
        print(f"Toys completed: {len(pulls)}, failed: {failed}")

        all_pulls[bin_idx] = {
        "pulls": pulls,
        "failed": failed,
        "fit_params": fit_params
        }

        # --- SAVE AFTER EACH BIN ---
        joblib.dump(all_pulls, "binned_pulls_main.pkl")
        print("Progress saved.")

        # --- RESET GRAPH after each bin ---
        

        zfit.run.clear_graph_cache()     # clears zfit computation graph
        gc.collect()   
        # tf.compat.v1.reset_default_graph()
        print("Graph reset.\n")

        # Plot
        # mean, std = plot_pulls(pulls, title=f"Q² bin {bin_idx}")
        # print(f"Pull mean={mean:.3f}, std={std:.3f}")

    exit()


from zfit_func_pulls import *
# high_conf_signal = high_conf_signal[(high_conf_signal['B_invariant_mass'] > 5000) & (high_conf_signal['B_invariant_mass'] < 5500)]
# sss = high_conf_signal[(high_conf_signal['B_invariant_mass'] < 5700) & (high_conf_signal['B_invariant_mass'] > 5150)].copy()
high_conf_signal_with_vetoes = post_selection_vetoes(high_conf_signal.copy())

A_raw_tot, A_raw_err_tot_stat, yield_p, yield_m, *_ = fit_asymmetry_cb(high_conf_signal_with_vetoes.copy())
print(f'Yields: Positive {yield_p} and negative {yield_m}')
main_pull = joblib.load("main_pulls.pkl")[0]["pulls"]
main_syst_err = np.sqrt((np.mean(main_pull) * A_raw_err_tot_stat) ** 2 + (A_raw_err_tot_stat * np.std(main_pull) / np.sqrt(len(main_pull))) ** 2)
A_raw_tot = A_raw_tot - jpsik_acp
syst_err = np.sqrt(main_syst_err ** 2 + jpsik_acp_err ** 2)
A_raw_err_tot = np.sqrt(A_raw_err_tot_stat ** 2 + syst_err ** 2)
print(f'Corrected CP Asymmetry (raw) for Kmumu decay is {A_raw_tot} +- {A_raw_err_tot} +- {A_raw_err_tot_stat} +- {syst_err}')

binned_data_all = calc_acp_q2_bins(high_conf_signal_with_vetoes, visual=True)
print(f"{'Acp':>7} | {'A_err':>7} | {'stat_err':>10} | {'sys_err':>12}")
for a, a_err, stat_err, sys_err in zip(binned_data_all['Acp'], binned_data_all['A_err'], binned_data_all['stat_err'], binned_data_all['sys_err']):
    print(f"{a:>7.4f} +- {a_err:>7.4f} +- {stat_err:>7.4f} +- {sys_err:>7.4f}")