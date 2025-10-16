from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
import re
import textwrap
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt


training_labels = ['Kaon_impact_parameter_chi2_wrt_primary_vertex', 'B_decay_vertex_fit_chi2',
                   'Kaon_PID_NN_score_for_muon_hypothesis', 'dimuon_system_flight_distance_wrt_B_decay_vertex',
                   'Isolation__B_vertex_delta_chi2_adding_two_extra_tracks__best_fits_', 'B_assumed_particle_type',
                   'Opposite_sign_muon_PID_NN_score_for_muon_hypothesis', 'Same_sign_muon_PID_NN_score_for_muon_hypothesis',
                   'Kaon_PID_NN_score_for_kaon_hypothesis', 'B_decay_vertex_x_position',
                   'B_cos_angle__between_line_of_flight_and_momentum', 'Isolation__B_mass_if_one_extra_track__best_fit__is_added',
                   'B_4_momentum_x_component', 'Isolation__B_vertex_delta_chi2_adding_one_extra_track__best_fit_',
                   'dimuon_system_impact_parameter_chi2_wrt_primary_vertex', 'B_impact_parameter_wrt_primary_vertex',
                   'dimuon_system_flight_distance_chi2_wrt_primary_ver\ntex', 'B_decay_vertex_y_position',
                   'dimuon_system_cos_angle__between_line_of_flight_from_primary_vertex_and_momentum',
                   'B_magnitude_of_momentum_transverse_to_beam']
training_labels = '|'.join(training_labels)
def train_model(data_sig, data_back, training_labels=training_labels, t_params=[30, 2000, 0.2, 6], rocauc=False, feat_imp=False, class_output=False):
    data_sig['target'] = 1
    data_back['target'] = 0
    data = pd.concat([data_sig, data_back], ignore_index=True)

    # Separate features and target
    X = data.drop('target', axis=1)
    X.columns = [re.sub(r'[^A-Za-z0-9_]', '_', c) for c in X.columns]
    # X = X.drop(columns=forbidden_vars)
    # X = X.drop(columns=X.filter(regex='^Isolation').columns)
    X = X.drop(columns=X.filter(regex=f'^(?!.*{training_labels}).+$'))
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001)

    lgbm = lgb.LGBMClassifier(early_stopping_round=t_params[0], n_estimators=t_params[1], learning_rate=t_params[2], max_depth=t_params[3])
    lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    y_pred = lgbm.predict_proba(X_test)[:,1]

    if rocauc:
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.show()

    if feat_imp:
        wrapped_cols = ["\n".join(textwrap.wrap(c, width=25)) for c in X_train.columns]
        X_train.columns = wrapped_cols
        # As a DataFrame for easier viewing
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'SplitImportance': lgbm.booster_.feature_importance(importance_type='split'),
            'GainImportance': lgbm.booster_.feature_importance(importance_type='gain')
        }).sort_values('GainImportance', ascending=False)
        fig, ax = plt.subplots()
        ax = lgb.plot_importance(lgbm.booster_, importance_type='gain', max_num_features=6, ax=ax)
        ax.set_yticklabels(feature_importance['Feature'].head(6)[::-1])
        plt.tight_layout()
        # plt.savefig('/Users/zifei/Desktop/feature_importance.png', dpi=300)
        plt.show()

    if class_output:
        y_pred_signal = y_pred[y_test == 1]
        y_pred_background = y_pred[y_test == 0]
        # Plot histograms
        plt.figure(figsize=(8,6))
        plt.hist(y_pred_background, bins=100, alpha=0.5, label='Background', density=True)
        plt.hist(y_pred_signal, bins=100, alpha=0.5, label='Signal', density=True)
        plt.xlabel('Classifier output (signal probability)')
        plt.ylabel('Normalized events')
        plt.title('Classifier response')
        plt.legend(loc='upper center')
        plt.grid(True)
        plt.yscale('log')
        plt.show()

    return lgbm, y_test, y_pred

def apply_model(data, model, threshold=0.985, training_labels=training_labels):
    data.columns = [re.sub(r'[^A-Za-z0-9_]', '_', c) for c in data.columns]
    data_test = data.copy()
    data_test = data_test.drop(columns=data.filter(regex=f'^(?!.*{training_labels}).+$'))
    preds = model.predict_proba(data_test)
    id_signal_data = data[preds[:,1] > threshold]
    return id_signal_data