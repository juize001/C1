import pickle
import pandas as pd
import matplotlib.pyplot as plt
import re
# with open('LHCb/rapidsim_JpsiK.pkl', 'rb') as infile:
#     data_jpsik = pickle.load(infile)
# with open('LHCb/rapidsim_Kmumu.pkl', 'rb') as infile:
#     data_kmumu = pickle.load(infile)
# with open('LHCb/dataset_2011.pkl', 'rb') as infile:
#     data_2011 = pickle.load(infile)
# with open('LHCb/samesign_2012.pkl', 'rb') as infile:
#     samesign_2011 = pickle.load(infile)

data_2012u = pd.read_pickle('LHCb/dataset_2012_MagnetUp.pkl')
data_2012d = pd.read_pickle('LHCb/dataset_2012_MagnetDown.pkl')
data_2012 = pd.concat([data_2012u, data_2012d])
# fs_data = pd.read_pickle("ProcessedData/first_selection_data.pkl")
# print(len(samesign_2011['B invariant mass']))

for item in list(data_2012d):
    print(item)
data_2012.columns = [re.sub(r'[^A-Za-z0-9_]', '_', c) for c in data_2012.columns]
# plt.hist(fs_data['dimuon_system_invariant_mass'], bins=1000)
# plt.hist(data_2012d['Magnet polarity'], bins=10)
# plt.hist(data_2012u['Magnet polarity'], bins=10)
plt.hist(data_2012['Kaon_PID_NN_score_for_muon_hypothesis'], bins=100)
plt.grid()
plt.show()