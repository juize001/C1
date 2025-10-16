import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import load_workbook
from openpyxl.formatting.rule import ColorScaleRule

def split_into_q2_bins(data, bin_bounds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12.5, 15, 17, 19, 23]):
    data['dimuon_mass_squared'] = (data['dimuon_system_invariant_mass'] / 1000) ** 2
    data["q2_bin"] = pd.cut(data["dimuon_mass_squared"], bins=bin_bounds, right=False, labels=False)

    # Separate DataFrames per bin
    binned_data = {}

    for i in range(len(bin_bounds) - 1):
        if bin_bounds[i] == 8:
            binned_resonant_data = data[data["q2_bin"] == i].copy()
        elif bin_bounds[i] == 12.5:
            continue
        else:
            binned_data[i] = data[data["q2_bin"] == i].copy()

    binned_resonant_data = binned_resonant_data[(abs(binned_resonant_data['B_invariant_mass'] - 5280) < 150) & (abs(binned_resonant_data['dimuon_system_invariant_mass'] - 3097) < 50)]
    return binned_data, binned_resonant_data, bin_bounds

def kmu_mass(k_E, mu_E, k_px, k_py, k_pz, mu_px, mu_py, mu_pz):
        E_tot = k_E + mu_E
        px_tot = k_px + mu_px
        py_tot = k_py + mu_py
        pz_tot = k_pz + mu_pz
        inv_mass = E_tot**2 - (px_tot**2 + py_tot**2 + pz_tot**2)
        return np.sqrt(np.maximum(inv_mass, 0.0))

def kmu_mass_filter(fs_data):
    fs_data = fs_data[(fs_data['B_invariant_mass'] < 5500) & (fs_data['B_invariant_mass'] > 5000)].copy()
    fs_data['kmu_mass'] = kmu_mass(fs_data['Kaon_4_momentum_energy_component'], fs_data['Opposite_sign_muon_4_momentum_energy_component'],
                                fs_data['Kaon_4_momentum_x_component'], fs_data['Kaon_4_momentum_y_component'],
                                fs_data['Kaon_4_momentum_z_component'], fs_data['Opposite_sign_muon_4_momentum_x_component'],
                                fs_data['Opposite_sign_muon_4_momentum_y_component'], fs_data['Opposite_sign_muon_4_momentum_z_component'])

    fs_data = fs_data[fs_data['Kaon_PID_NN_score_for_kaon_hypothesis'] > 0.4]
    fs_data = fs_data[(fs_data['kmu_mass'] < 1840) | (fs_data['kmu_mass'] > 1890)]
    return fs_data

def acp_calc(pd1, pd2):
    n1 = len(pd1)
    n2 = len(pd2)
    acp = (n1 - n2) / (n1 + n2)
    err = 2 / (n2 + n1)**2 * np.sqrt((n1 * np.sqrt(n2))**2 + (n2 * np.sqrt(n1))**2)
    return acp, err

def get_val_err(result, param):
        info = result.params[param]
        val = info["value"]
        err = (
            info.get("error")
            or (info.get("hesse") or {}).get("error")
            or 0.0
        )
        return val, err

def colour(dir, o_dir):
    df = pd.read_excel(dir, index_col=0)
    wb = load_workbook(dir)
    ws = wb.active   # assuming the first sheet

    n_rows = df.shape[0]
    n_cols = df.shape[1]

    start_cell = ws.cell(row=2, column=2).coordinate
    end_cell = ws.cell(row=1 + n_rows, column=1 + n_cols).coordinate
    cell_range = f"{start_cell}:{end_cell}"

    color_scale = ColorScaleRule(
        start_type='num', start_value=0, start_color='ffffff',   # green
        end_type='num',   end_value=1, end_color='FF0000'        # red
    )
    ws.conditional_formatting.add(cell_range, color_scale)
    # Save the workbook
    wb.save(o_dir)

# fs_data = pd.read_pickle("/Users/zifei/Desktop/C1/ProcessedData/first_selection_data2.pkl")
# plt.hist((kmu_mass_filter(fs_data)['kmu_mass']), bins=50)
# plt.grid()
# # plt.savefig('/Users/zifei/Desktop/kmu_minus.png', dpi=300)
# plt.show()